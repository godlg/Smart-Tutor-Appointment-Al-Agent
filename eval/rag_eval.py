"""RAG 评估主体: 真检索链 (HybridVectorStore) + 多指标 + BM25 对照。

从 eval/runner.py 抽出来独立成模块 (runner 只保留编排):
    runner._evaluate_rag → evaluate_rag(冻结集 dict, k)

评估口径 (指标全部进 report.results.rag.metrics):
    recall_at_k / mrr            复用 RecallAtKEvaluator (与 v1 同口径, 可比)
    ndcg_at_k                    排序质量 (命中越靠前越高)
    empty_recall_rate            空召回率 (recall == 0 的 case 占比)
    bm25_baseline_recall_at_k    同一冻结集上 BM25 单路的 Recall@k (对照, 不入门禁)
    bm25_baseline_mrr            同上, MRR
    embedding_mode               "bow_stub" (离线词袋桩) | "live" (真实 embedding)
    rerank_active                本轮是否真的跑过精排 (模型缺失回退 RRF 时如实标 False)
    rerank_threshold             实际生效的精排阈值; 关闭(默认)时为 None → JSON 里的 null
    case_count                   参与评估的 case 数

    ⚠️ `bow_stub` 不是语义模型: 它只反映**词重合** (jieba 分词 → 词哈希 → 子线性 tf →
    L2 归一化), 近义改写 / 跨语言 / 指代一概不会。真实语义数值用 EVAL_RAG_LIVE=1 跑。

三个"答不上来就别装绿"的守卫 (任一不过 → 抛 ValueError → 门禁红):
    1. 反逐字改写: query 与语料任一篇的最长公共子串必须 < 12 字;
    2. 悬空期望: expected_doc_ids 必须都能在 corpus 里找到;
    3. ground truth 可达性: 每条 case 的期望 doc 至少有一个能被「文档自身内容」
       作为 query 检索到 —— 否则这条 case 在测试的检索配置下无解, 属于坏标签。

**链路用生产默认配置** (`HybridVectorStore` → `HybridRetriever` 读 env
RERANK_ENABLED / RAG_MIN_RERANK_SCORE), 评估器不覆盖这两个开关 ——
门禁要测的就是生产链路本身, 包括它的精排相关性阈值 (默认 None = 不过滤)。
"""


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import empty_recall_rate, mean, ndcg_at_k
from .rag_guard import (
    corpus_doc_ids_of,
    corpus_texts_of,
    expected_ids_dangling,
    format_violations,
    guard_cases,
    max_common_substring_len,
)
from .rag_store import (
    build_corpus_store,
    live_mode,
    patched_embedding,
    query_transform_variants,
)

# 精排候选深度 (与 HybridRetriever 默认 candidates 一致)
DEFAULT_K = 3


# ----------------------------------------------------------------------
# 检索执行 (计时 + 异常隔离 + 记录排名)
# ----------------------------------------------------------------------

class _SearchRecorder:
    """包一层 store: 记录每 case 的命中 id 列表与耗时, 再交给 RecallAtKEvaluator。

    这样召回/ MRR 仍由 `RecallAtKEvaluator` 算出 (口径与 v1 一致), NDCG 用这里
    记录的排名算 —— 每条 query 只检索一次 (精排很贵, 不能为了指标跑两遍)。
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self.ranked: List[List[int]] = []
        self.latencies: List[float] = []
        self.errors: List[Optional[str]] = []

    def search(self, query: str, top_k: int = DEFAULT_K) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        error: Optional[str] = None
        try:
            results = self._store.search(query, top_k)
        except Exception as exc:  # noqa: BLE001 - 单 case 检索失败记 error, 不中断整体
            results = []
            error = f"{type(exc).__name__}: {exc}"
        self.latencies.append((time.perf_counter() - t0) * 1000.0)
        self.errors.append(error)
        self.ranked.append([int(r["id"]) for r in results if r.get("id") is not None])
        return results


def _score_store(
    store: Any, cases: Sequence[Dict[str, Any]], k: int
) -> Dict[str, Any]:
    """在一个 store 上跑完所有 case, 返回 (evaluator 输出, 排名, 耗时, 异常)。"""
    from services.components.base import QueryCase
    from services.components.evaluators import RecallAtKEvaluator

    recorder = _SearchRecorder(store)
    evaluator = RecallAtKEvaluator(
        [QueryCase(query=c["query"], expected_doc_ids=list(c["expected_doc_ids"])) for c in cases],
        k=k,
    )
    out = evaluator.evaluate(recorder)
    return {
        "evaluator": out,
        "ranked": recorder.ranked,
        "latencies": recorder.latencies,
        "errors": recorder.errors,
    }


def _ndcg_of(scored: Dict[str, Any], cases: Sequence[Dict[str, Any]], k: int) -> float:
    return mean([
        ndcg_at_k(scored["ranked"][i], c["expected_doc_ids"], k)
        for i, c in enumerate(cases)
    ])


# ----------------------------------------------------------------------
# Ground truth 可达性
# ----------------------------------------------------------------------

def check_answerable(
    corpus: Sequence[Dict[str, Any]], cases: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """检查每条 case 的期望 doc 是否「可被检索到」。

    做法: 生产 BM25 路用的就是 `Bm25Index` (content + keywords)。拿每个期望 doc 的
    **自身内容**当 query 去 BM25 检索, 命中它自己即说明该标签在测试配置下可解。
    期望 doc 全都不可达 → 这条 case 是坏标签 (检索器再强也拿不到), 直接报错。

    注: 这里只用 BM25 做**可达性**判定, 不参与任何门禁指标; 它恰好也是
    「生产 BM25 路径能不能看见这条 doc」的检查 (候选集在 hybrid 里先按
    embedding 非空过滤, 建集脚本采样的 chunk 都带向量)。
    """
    from services.bm25_index import Bm25Index

    docs = [{"id": int(d["doc_id"]), "content": d["content"], "keywords": d.get("keywords") or []}
            for d in corpus]
    by_id = {d["id"]: d for d in docs}
    index = Bm25Index().build(docs)

    problems: List[Dict[str, Any]] = []
    for case in cases:
        reachable = []
        for doc_id in case["expected_doc_ids"]:
            doc = by_id.get(int(doc_id))
            if doc is None:
                continue
            hits = [did for did, _rank, _score in index.search(doc["content"], top_n=5)]
            if int(doc_id) in hits:
                reachable.append(int(doc_id))
        if not reachable:
            problems.append({
                "id": case["id"],
                "expected": list(case["expected_doc_ids"]),
                "reason": "期望文档在测试检索配置下不可达 (自身内容都检不到自己)",
            })
    return problems


def format_answerable_problems(problems: Sequence[Dict[str, Any]]) -> str:
    lines = [f"ground truth 可达性检查未通过: {len(problems)} 条 case 无解:"]
    for p in problems:
        lines.append(f"  - {p['id']}: 期望 {p['expected']} | {p['reason']}")
    lines.append("  → 请更换期望 doc, 或修正语料 (坏标签会让门禁永远红)。")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 主体
# ----------------------------------------------------------------------

def evaluate_rag(data: Dict[str, Any], k: int = DEFAULT_K) -> Dict[str, Any]:
    """跑一条完整的 RAG 评估, 返回 {"details", "metrics", "diagnostics"}。"""
    corpus = data.get("corpus") or []
    cases = data.get("cases") or []
    if not corpus or not cases:
        raise ValueError("rag 评估集缺少 corpus 或 cases")

    # ---- 守卫 1: 反逐字改写 ----
    violations = guard_cases(cases, corpus_texts_of(data))
    if violations:
        raise ValueError(format_violations(violations))

    # ---- 守卫 2: 期望 doc 无悬空 ----
    dangling = expected_ids_dangling(cases, corpus_doc_ids_of(data))
    if dangling:
        raise ValueError(
            "rag 评估集期望悬空 (expected_doc_ids 不在 corpus 里): "
            + "; ".join(f"{d['id']} → {d['missing']}" for d in dangling)
        )

    # ---- 守卫 3: 期望 doc 可达 ----
    unreachable = check_answerable(corpus, cases)
    if unreachable:
        raise ValueError(format_answerable_problems(unreachable))

    live = live_mode()
    from services.components.vector_stores import Bm25VectorStore
    from services import reranker

    with patched_embedding():
        store, docs = build_corpus_store(corpus)
        retriever = store._retriever  # HybridVectorStore 内部即 HybridRetriever
        rerank_active = bool(getattr(retriever, "rerank", False) and reranker.available())
        # 实际生效的精排阈值: None = 不设阈值(不过滤); 见 hybrid_retriever 的实测说明
        rerank_threshold = getattr(retriever, "min_rerank_score", None)
        search_store = _live_search_store(retriever) if live else store

        hybrid = _score_store(search_store, cases, k)

        # BM25 单路对照 (同一冻结集; 不入门禁)
        bm25_store = Bm25VectorStore()
        bm25_store.add(docs)
        bm25 = _score_store(bm25_store, cases, k)

    details = _build_details(cases, hybrid, k)
    metrics = {
        "recall_at_k": float(hybrid["evaluator"].get("recall_at_k") or 0.0),
        "mrr": float(hybrid["evaluator"].get("mrr") or 0.0),
        "ndcg_at_k": _ndcg_of(hybrid, cases, k),
        "empty_recall_rate": empty_recall_rate(hybrid["evaluator"].get("case_results", [])),
        "bm25_baseline_recall_at_k": float(bm25["evaluator"].get("recall_at_k") or 0.0),
        "bm25_baseline_mrr": float(bm25["evaluator"].get("mrr") or 0.0),
        # 词袋桩: 只反映词重合, **不代表真实 embedding 的语义召回能力**
        # (近义改写 / 跨语言 / 指代一概不会)。真实语义数值用 EVAL_RAG_LIVE=1。
        "embedding_mode": "live" if live else "bow_stub",
        "rerank_active": rerank_active,
        # 实际生效的精排阈值; 关闭(默认)时为 None → JSON 里是 null
        "rerank_threshold": rerank_threshold,
        "case_count": len(cases),
    }
    diagnostics = {
        "k": k,
        "corpus_size": len(corpus),
        "guard_max_common": max(
            (max_common_substring_len(c["query"], corpus_texts_of(data)) for c in cases),
            default=0,
        ),
        "bm25_baseline_ndcg_at_k": _ndcg_of(bm25, cases, k),
        "bm25_baseline_empty_recall_rate": empty_recall_rate(
            bm25["evaluator"].get("case_results", [])
        ),
        "rerank_model": reranker.active_model() if rerank_active else None,
        "query_transform": "on" if live else "off",
        "embedding_note": (
            "词袋桩: 只反映词重合, 不代表真实 embedding 的语义召回能力; "
            "真实数值用 EVAL_RAG_LIVE=1"
        ),
    }
    return {"details": details, "metrics": metrics, "diagnostics": diagnostics}


def _live_search_store(retriever: Any) -> Any:
    """live 模式: query 经 query 变换展开成多条, 走 search_multi (RRF + 精排)。"""

    class _MultiQueryStore:
        def search(self, query: str, top_k: int = DEFAULT_K) -> List[Dict[str, Any]]:
            variants = query_transform_variants(query)
            return _run_async(retriever.search_multi(variants, top_k=top_k))

    return _MultiQueryStore()


def _run_async(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _build_details(
    cases: Sequence[Dict[str, Any]], scored: Dict[str, Any], k: int
) -> List[Dict[str, Any]]:
    """逐 case 明细 (供人工排查: 期望 / 实际命中 / 排名 / 三指标)。"""
    case_results = scored["evaluator"].get("case_results", [])
    details: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        cr = case_results[i] if i < len(case_results) else {}
        recall = float(cr.get("recall", 0.0))
        ranked = scored["ranked"][i] if i < len(scored["ranked"]) else []
        error = scored["errors"][i] if i < len(scored["errors"]) else None
        detail: Dict[str, Any] = {
            "id": case["id"],
            "ok": recall >= 1.0,
            "latency_ms": round(scored["latencies"][i], 2) if i < len(scored["latencies"]) else 0.0,
            "tokens": 0,
            "steps": 0,
            "expected": list(case["expected_doc_ids"]),
            "actual": ranked,
            "recall": recall,
            "mrr": float(cr.get("mrr", 0.0)),
            "ndcg": ndcg_at_k(ranked, case["expected_doc_ids"], k),
        }
        if error:
            detail["error"] = error
        details.append(detail)
    return details
