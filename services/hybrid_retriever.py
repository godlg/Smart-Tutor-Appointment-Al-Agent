# services/hybrid_retriever.py
"""混合检索编排器: Dense(FAISS) + BM25 → RRF 融合 → ms-marco 精排。

检索五阶段(Query Trace 埋点):
    1. 候选集(全量 或 filter-then-vector 的 SQL 过滤子集)
    2. Dense 路: FAISS 内积检索
    3. BM25 路: jieba 分词 + BM25Okapi
    4. RRF 融合: score = Σ 1/(k + rank)
    5. 精排: ms-marco cross-encoder 打分(可开关; 失败自动回退 RRF 结果)

三级降级(P1-1): dense+BM25 → BM25-only → 空。
Dense 依赖外部 embedding API, 429/超时不应把同一轮已算好的 BM25 结果一起丢掉:
_dense_search 包在 try/except 里, 失败则 dense_ranked=[], RRF 退化为单路继续。
两路都失败/都无命中才返回 [], 由上层既有「未查到」降级文案接手。
实际走到哪一级记在 trace span.attrs["degraded"] 上(便于压测/排查)。

精排后处理(P1-2):
- 阈值: 只在本轮**真的跑过精排**时丢弃 rerank_score < min_rerank_score 的候选;
  RRF 兜底路径不应用(rrf_score 是 1/(k+rank) 的小正数, 阈值会误杀全部结果)。
- 去重: 按规范化内容(去空白+小写)与 (source_hash, chunk_index) 双重键, 保留排名靠前的那条。

多查询(search_multi): 候选集与 BM25 索引各建一次, 每条 query 各跑一次 dense,
全部 [(doc_id, rank)] 一起 RRF 融合后再精排/阈值/去重。
search() 是 search_multi([q]) 的薄包装, 单查询行为与改动前一致。

与 KnowledgeService.search()(纯 Dense) 完全解耦: 不修改原检索路径,
由调用方(knowledge_retriever)按 RETRIEVAL_MODE 选择 dense / hybrid。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .bm25_index import Bm25Index
from . import reranker
from .text_embedding import embed_input
from observability import trace_span

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def rerank_enabled_default() -> bool:
    """精排开关默认开(模型已随仓库 cache/ 提供)。"""
    return _env_bool("RERANK_ENABLED", True)


def min_rerank_score_default() -> Optional[float]:
    """精排相关性阈值默认 **None = 不设阈值 (不过滤)**; env RAG_MIN_RERANK_SCORE 显式设置才生效。

    为什么默认关掉 0.0 (2026-10, spec delta 2 §0-B 实测):
        旧注释曾写「bge-reranker-base 实测相关 4~6 分 / 不相关全为负, 0.0 是天然分界」——
        那个经验来自**逐字改写**的 trivial query, 分布不可推广。改用口语化 query 后
        对 24 条 query × 34 篇 corpus 做纯精排全量打分:

            阈值关闭(-999) → recall@3 = 0.9167 (22/24), 空召回率 0.0833
            阈值 0.0       → recall@3 = 0.5000        , 空召回率 0.5000

        12/24 条 case 的期望 doc 精排分为**负**, 且多数还是排名第 1 的正确答案
        (rag_008: 排名 1 / 分 -4.8; rag_007: 排名 3 / 分 -3.96;
         rag_002: 排名 1 / 分 -1.53; rag_011: 排名 1 / 分 -0.16)。
        结论: bge 的 logit 随 query 口语化程度**整体下移**, 0.0 不是中性分界,
        当默认值等于把半数正确答案丢掉 → 该默认值必须撤。

    机制保留: 传了数值就照数值过滤 (便于按模型量纲做实验 / 线上按需收紧)。
    非法值告警并回退 None (不让配置错误炸掉检索, 也不静默启用一个没校准的阈值)。
    """
    raw = os.getenv("RAG_MIN_RERANK_SCORE")
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("RAG_MIN_RERANK_SCORE=%r 非法, 回退 None (不设阈值)", raw)
        return None


def _dedup_keys(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """一条候选的全部去重键:
    - ("chunk", "source_hash:chunk_index"): 同一文件同一块视为同一条
    - ("content", 规范化内容): 去掉全部空白后转小写, 用于跨来源的重复内容
    任一键命中已有结果即判定为重复。"""
    keys: List[Tuple[str, str]] = []
    source_hash = doc.get("source_hash")
    chunk_index = doc.get("chunk_index")
    if source_hash is not None and chunk_index is not None:
        keys.append(("chunk", f"{source_hash}:{chunk_index}"))
    normalized = "".join((doc.get("content") or "").split()).lower()
    if normalized:
        keys.append(("content", normalized))
    return keys


def dedup_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 规范化内容 / (source_hash, chunk_index) 去重, 保留排名靠前的一条。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for doc in docs:
        keys = _dedup_keys(doc)
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(doc)
    return out


def rrf_merge(
    ranked_lists: List[List[Tuple[int, int]]], k: int = 60
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion。

    ranked_lists: 各路结果 [(doc_id, rank)], rank 从 1 开始;
    返回 [(doc_id, rrf_score)] 按分数降序。k 常用 60。
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """双路召回 + RRF + 精排。依赖注入 KnowledgeService(便于测试)。"""

    def __init__(
        self,
        knowledge_service: Any,
        rrf_k: int = 60,
        candidates: int = 20,
        rerank: Optional[bool] = None,
        min_rerank_score: Optional[float] = None,
    ) -> None:
        self.ks = knowledge_service
        self.rrf_k = rrf_k
        self.candidates = candidates
        # None → 走环境变量默认(生产 hybrid 全开); 测试可显式关
        self.rerank = rerank if rerank is not None else rerank_enabled_default()
        # None → 走环境变量默认 (默认即 None = 不过滤); 只在精排真跑过且非 None 时生效
        self.min_rerank_score = (
            min_rerank_score if min_rerank_score is not None
            else min_rerank_score_default()
        )

    async def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """单查询混合检索(薄包装: 等价于 search_multi([query]))。"""
        return await self.search_multi(
            [query], top_k=top_k, category=category,
            subject=subject, grade_level=grade_level, doc_type=doc_type,
        )

    async def search_multi(
        self,
        queries: List[str],
        top_k: int = 3,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """多查询混合检索: 候选集与 BM25 索引各建一次, 每条 query 各跑一次 dense,
        全部 [(doc_id, rank)] 一起丢进 RRF 融合, 精排/阈值/去重只做一次。

        queries 为空时按单条空串处理(行为等价于旧的 search(""))。
        """
        if not queries:
            queries = [""]
        with trace_span(
            "rag_search",
            mode="hybrid",
            query=queries[0][:30],
            top_k=top_k,
            has_meta_filter=any(v is not None for v in (subject, grade_level, doc_type)),
        ) as span:
            return await self._search_multi_impl(
                queries=queries, top_k=top_k, category=category,
                subject=subject, grade_level=grade_level, doc_type=doc_type,
                span=span,
            )

    async def _search_impl(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        subject: Optional[str],
        grade_level: Optional[str],
        doc_type: Optional[str],
        span=None,
    ) -> List[Dict[str, Any]]:
        """单查询检索主体(保留旧签名, 供既有调用方/演示脚本直接驱动)。"""
        return await self._search_multi_impl(
            queries=[query], top_k=top_k, category=category,
            subject=subject, grade_level=grade_level, doc_type=doc_type,
            span=span,
        )

    @staticmethod
    def _query_degraded(
        dense_ranked: List[Tuple[int, int]],
        bm25_ranked: List[Tuple[int, int]],
        dense_failed: bool,
        bm25_failed: bool,
    ) -> str:
        """单条 query 的降级标签(与单查询版语义一致)。"""
        if not dense_ranked and not bm25_ranked:
            return "empty" if (dense_failed or bm25_failed) else "no_hit"
        if dense_failed or not dense_ranked:
            return "bm25_only"
        if bm25_failed or not bm25_ranked:
            return "dense_only"
        return "none"

    @staticmethod
    def _aggregate_degraded(per_query: List[str], any_exception: bool) -> str:
        """多路降级聚合: 全空(无异常)→ no_hit;全空(有过异常)→ empty;
        至少一条 query 只剩 BM25 → bm25_only;至少一条只剩 dense → dense_only;否则 none。"""
        if all(label in ("no_hit", "empty") for label in per_query):
            return "empty" if any_exception else "no_hit"
        if any(label == "bm25_only" for label in per_query):
            return "bm25_only"
        if any(label == "dense_only" for label in per_query):
            return "dense_only"
        return "none"

    async def _search_multi_impl(
        self,
        queries: List[str],
        top_k: int,
        category: Optional[str],
        subject: Optional[str],
        grade_level: Optional[str],
        doc_type: Optional[str],
        span=None,
    ) -> List[Dict[str, Any]]:
        """混合检索主体(多 query;与 search 分离以便 span attrs 结束时统一挂载)。"""
        stages: Dict[str, float] = {}
        t0 = time.perf_counter()

        # ---- 阶段1: 候选集(filter-then-vector 的 SQL 过滤在此发生, 多查询只做一次) ----
        has_meta_filter = any(
            v is not None for v in (subject, grade_level, doc_type)
        )
        if has_meta_filter:
            candidates = self.ks.db.search_documents_with_filters(
                subject=subject, grade_level=grade_level, doc_type=doc_type,
            )
        else:
            candidates = self.ks.db.get_all_documents()
        # Dense/BM25 两路必须同一候选集: 只保留有 embedding 的(与 FAISS 索引对齐)
        candidates = [
            c for c in candidates
            if c.get("embedding") and c.get("content")
        ]
        if not candidates:
            if span is not None:
                span.attrs["degraded"] = "no_candidates"
                span.attrs["result_count"] = 0
                span.attrs["variants"] = len(queries)
            return []
        doc_map: Dict[int, Dict[str, Any]] = {int(c["id"]): c for c in candidates}
        stages["candidates"] = time.perf_counter() - t0

        # ---- 阶段2: Dense 路(FAISS 内积, 与现有 IndexFlatIP 一致; 每条 query 各一次) ----
        # embedding API 是外部依赖, 失败不应连累同一轮已算好的 BM25 结果 → 降级为空路;
        # 单条 query 失败也不连累其它 query(各 query 独立 try)。
        t1 = time.perf_counter()
        dense_lists: List[List[Tuple[int, int]]] = []
        dense_failed_flags: List[bool] = []
        for query in queries:
            try:
                dense_lists.append(self._dense_search(query, candidates))
                dense_failed_flags.append(False)
            except Exception as e:  # noqa: BLE001 - 单路失败降级, 不冒泡
                logger.warning(f"dense 路检索失败, 降级为 BM25-only: {e}")
                dense_lists.append([])
                dense_failed_flags.append(True)
        stages["dense"] = time.perf_counter() - t1

        # ---- 阶段3: BM25 路(索引只建一次, 每条 query 各查一次) ----
        # 两路都失败/都无命中 → stage4 的 merged_docs 为空 → 返回 [], 由上层
        # 「未查到」降级文案接手。这里同样不让单路异常冒泡。
        t2 = time.perf_counter()
        bm25_failed = False
        bm25_lists: List[List[Tuple[int, int]]] = []
        try:
            bm25 = Bm25Index().build(candidates)
            for query in queries:
                bm25_lists.append([
                    (doc_id, rank)
                    for doc_id, rank, _ in bm25.search(query, self.candidates)
                ])
        except Exception as e:  # noqa: BLE001 - 单路失败降级, 不冒泡
            logger.warning(f"BM25 路检索失败, 降级为 dense-only: {e}")
            bm25_lists = [[] for _ in queries]
            bm25_failed = True
        stages["bm25"] = time.perf_counter() - t2

        # 降级标记: 先按 query 算, 再聚合
        degraded_per_query = [
            self._query_degraded(
                dense_lists[i], bm25_lists[i],
                dense_failed_flags[i], bm25_failed,
            )
            for i in range(len(queries))
        ]
        any_exception = bm25_failed or any(dense_failed_flags)
        degraded = self._aggregate_degraded(degraded_per_query, any_exception)

        # ---- 阶段4: RRF 融合(query × path 的全部排名一起融合) ----
        t3 = time.perf_counter()
        merged = rrf_merge(
            [lst for lst in dense_lists] + [lst for lst in bm25_lists],
            self.rrf_k,
        )[: self.candidates]
        merged_docs: List[Dict[str, Any]] = []
        for doc_id, rrf_score in merged:
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            item = dict(doc)
            item["rrf_score"] = rrf_score
            merged_docs.append(item)
        stages["rrf"] = time.perf_counter() - t3

        # ---- 阶段5: 精排(只在融合后的候选上跑一次; 失败回退 RRF) ----
        t4 = time.perf_counter()
        reranked: Optional[List[Dict[str, Any]]] = None
        if self.rerank and reranker.available():
            try:
                reranked = reranker.rerank(queries[0], merged_docs, top_k=top_k)
            except Exception as e:
                logger.warning(f"精排失败, 回退 RRF 结果: {e}")
                reranked = None
        rerank_applied = reranked is not None
        if not rerank_applied:
            reranked = merged_docs[:top_k]
        stages["rerank"] = time.perf_counter() - t4

        # ---- 阈值: 只在精排真的跑过**且**显式配了阈值时生效 ----
        # (RRF 分数量纲与精排分不同, 兜底路径一律不套阈值;
        #  min_rerank_score 为 None = 不设阈值, 见 min_rerank_score_default 的实测说明)
        if rerank_applied and self.min_rerank_score is not None:
            reranked = [
                d for d in reranked
                if d.get("rerank_score", 0.0) >= self.min_rerank_score
            ]

        # ---- 结果封装(score = 精排分, 否则 rrf 融合分) ----
        filtered: List[Dict[str, Any]] = []
        for doc in reranked:
            if category and doc.get("category") != category:
                continue
            filtered.append(doc)

        # 去重(保排名靠前的), 之后重排 rank 保证连续
        results: List[Dict[str, Any]] = []
        for rank, doc in enumerate(dedup_docs(filtered), start=1):
            item = dict(doc)
            item["score"] = doc.get("rerank_score", doc.get("rrf_score", 0.0))
            item["rank"] = rank
            results.append(item)

        logger.info(
            "hybrid_search query=%r variants=%d stages=%s degraded=%s took=%.1fms top=%d/%d",
            queries[0][:30], len(queries), stages, degraded,
            (time.perf_counter() - t0) * 1000, len(results), len(candidates),
        )
        # Trace 埋点:把五阶段耗时挂到 rag_search span attrs 上
        if span is not None:
            span.attrs["stages_ms"] = {
                k: round(v * 1000, 2) for k, v in stages.items()
            }
            span.attrs["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            span.attrs["result_count"] = len(results)
            span.attrs["degraded"] = degraded
            span.attrs["variants"] = len(queries)
            span.attrs["degraded_per_query"] = degraded_per_query
        return results

    def _dense_search(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """候选集上 FAISS 检索, 返回 [(doc_id, rank)]。"""
        embs = np.array(
            [c["embedding"] for c in candidates], dtype="float32"
        )
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        query_emb = np.array([embed_input(query)], dtype="float32")
        k = min(self.candidates, len(candidates))
        scores, indices = index.search(query_emb, k)
        ranked: List[Tuple[int, int]] = []
        for rank, idx in enumerate(indices[0], start=1):
            if 0 <= idx < len(candidates):
                ranked.append((int(candidates[idx]["id"]), rank))
        return ranked
