"""RAG 评估门禁修复 (spec: rag-eval-gate-fix) 的测试。

覆盖:
1. 无 LLM 凭据也能跑 `--only rag` (P0-2)
2. rag 评估走真检索链 HybridVectorStore (不是 Bm25VectorStore) (P0-1)
3. metrics 里新指标齐备且为 float (P0-4)
4. 同一冻结集连跑两次指标逐位相等 (确定性)
5. BM25 对照指标存在 (P0-1)
6. 反逐字改写守卫: 抄原文 >= 12 字的 query 被拦下 (P0-3)
7. rag.json 期望对齐 + 规模下限 (P0-3)
8. 门禁负例: 期望 doc_id 全改成不存在 → main 返回 1 (P0-4)
9. 门禁正例: 正常冻结集 → main 返回 0, report 的 rag 段含新指标

**载重测试用缩小语料**: 真实的 34 篇 × bge 精排每 case ~19s, 全量一轮 > 8 分钟,
不适合单测。第 1/2/3/4/5/8/9 例把 `_sub_cases` 换成 3 篇小语料 + 3 条 case,
断言的是**链路结构 / 指标齐备 / 判门禁行为**; 全量语料由报告里的实跑证据覆盖。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval.rag_eval as rag_eval_module  # noqa: E402
from eval.metrics import dcg_at_k, empty_recall_rate, ideal_dcg_at_k, ndcg_at_k  # noqa: E402
from eval.rag_guard import (  # noqa: E402
    MIN_COMMON_SUBSTRING,
    guard_cases,
    longest_common_substring_len,
    max_common_substring_len,
)
from eval.rag_store import stub_embed  # noqa: E402

METRIC_KEYS = (
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "empty_recall_rate",
    "bm25_baseline_recall_at_k",
    "bm25_baseline_mrr",
)


# ----------------------------------------------------------------------
# 夹具: 缩小语料 (链路结构不变, 只是不跑 34 篇 × 精排)
# ----------------------------------------------------------------------

SUB_CORPUS = [
    {
        "doc_id": 101,
        "content": "初中数学 一元二次方程：判别式、根与系数关系(韦达定理)、动点问题等高频考点。",
        "keywords": ["初中", "数学", "韦达定理"],
        "subject": "数学", "grade_level": "初中", "doc_type": "subject_method",
        "category": "subject_method", "source_file": None,
    },
    {
        "doc_id": 102,
        "content": "小学阅读启蒙课：以经典绘本为载体，词汇 + 造句 + 复述故事。",
        "keywords": ["小学", "阅读", "绘本"],
        "subject": "语文", "grade_level": "小学", "doc_type": "course_intro",
        "category": "course_intro", "source_file": None,
    },
    {
        "doc_id": 103,
        "content": "高中生升学志愿填报：结合成绩位次与专业倾向，分冲稳保三档给出建议。",
        "keywords": ["高中", "志愿", "升学"],
        "subject": None, "grade_level": "高中", "doc_type": "parent_guide",
        "category": "parent_guide", "source_file": None,
    },
]

# 三条 case 都必须在真链路 (bge 精排 + 默认阈值 0.0) 下可召回 —— 否则
# 「门禁正例」测的就不是门禁而是精排的脾气。query 刻意保持口语化且与语料
# 无 >= 12 字公共子串 (守卫会拦), 但仍保留足以命中的关键词。
SUB_CASES = [
    {"id": "sub_001", "version": "2.0.0", "query": "二次方程判别式这类题怎么做",
     "expected_doc_ids": [101], "note": "n", "source_quote": "q", "label_source": "human_reviewed"},
    {"id": "sub_002", "version": "2.0.0", "query": "小学阅读课用绘本教什么内容",
     "expected_doc_ids": [102], "note": "n", "source_quote": "q", "label_source": "human_reviewed"},
    {"id": "sub_003", "version": "2.0.0", "query": "高考志愿填报分几档给建议",
     "expected_doc_ids": [103], "note": "n", "source_quote": "q", "label_source": "human_reviewed"},
]

SUB_DATASET = {"version": "2.0.0", "k": 3, "corpus": SUB_CORPUS, "cases": SUB_CASES}


# 真实实现 (import 时抓一次: 夹具会把 rag_eval_module.evaluate_rag 换成缩小语料版)
_REAL_EVALUATE_RAG = rag_eval_module.evaluate_rag


def _real_eval(data, k=3):
    """调用真正的 evaluate_rag 实现 (绕开夹具的替换)。"""
    return _REAL_EVALUATE_RAG(data, k)


def _run(coro):
    """同步跑 async 检索 (仓库未装 pytest-asyncio, 与其它测试同款)。"""
    import asyncio

    return asyncio.run(coro)


def _cos(a, b) -> float:
    """余弦相似度 (词袋桩已 L2 归一化, 内积即 cos)。"""
    return float(sum(x * y for x, y in zip(a, b)))


@pytest.fixture
def patched_embedding_env():
    """把词袋桩 patch 到 import 点 (与评测链路同款), 供直接驱 HybridRetriever 的用例。

    不 patch 的话 `KnowledgeService.initialize()` 会真去调远程 embedding API
    (实测会 ConnectError 挂掉), 这些用例必须离线跑。
    """
    from eval.rag_store import patched_embedding

    with patched_embedding():
        yield


def _requested_all_kinds():
    """六类全跑时的 requested_kinds (与 runner 默认一致)。"""
    from eval.runner import EDDRunner

    return sorted(EDDRunner(dry_run=True).only)


@pytest.fixture
def sub_dataset(monkeypatch):
    """让 `_evaluate_rag`(经 eval.runner 调用的那个 evaluate_rag) 读缩小冻结集。

    直接给 `eval.rag_eval.evaluate_rag` 打桩, 但桩里调的是 import 时抓到的
    真实实现 —— 因此链路结构 / 指标 / 守卫全都是真的, 只是语料变小。
    磁盘上的 `eval/cases/rag.json` 一个字节都不动。
    """
    monkeypatch.setattr(
        rag_eval_module, "evaluate_rag",
        lambda data, k=3: _REAL_EVALUATE_RAG(SUB_DATASET, k),
    )
    return SUB_DATASET


# ----------------------------------------------------------------------
# 1. 无 LLM 凭据也能跑 --only rag
# ----------------------------------------------------------------------

class TestNoLlmRequired:
    def test_only_rag_runs_without_llm_credentials(self, monkeypatch, tmp_path, sub_dataset):
        """create_chat_model 抛错时 `--only rag` 仍要跑完 (不抛 OpenAIError)。"""
        import eval.runner as runner_module

        def _boom(*_a, **_k):
            raise RuntimeError("没有 LLM 凭据 (模拟 OpenAIError)")

        monkeypatch.setattr("config.model_provider.create_chat_model", _boom, raising=False)
        monkeypatch.setattr(runner_module, "DEFAULT_REPORT", tmp_path / "report.json")

        # 只跑 rag: 门禁结果决定 exit code (0/1 都算跑通, 关键是不抛凭据错)
        code = runner_module.main(["--only", "rag", "--out", str(tmp_path / "report.json")])
        assert code in (0, 1), "只跑 rag 时不应因缺少 LLM 凭据而炸掉"

    def test_needs_llm_flag(self):
        from eval.runner import _needs_llm

        assert _needs_llm(["rag"]) is False
        assert _needs_llm(["rag", "rag"]) is False
        assert _needs_llm(["rag", "routing"]) is True
        assert _needs_llm(None) is True  # 全量: 需要 LLM


# ----------------------------------------------------------------------
# 2. 走真检索链
# ----------------------------------------------------------------------

class TestRealChain:
    def test_uses_hybrid_vector_store_not_bm25(self, monkeypatch, sub_dataset):
        """rag 评估必须构造 HybridVectorStore (真链路), 不是 Bm25VectorStore。"""
        import services.components.vector_stores as vs

        constructed = []
        original_hybrid = vs.HybridVectorStore

        class _SpyHybrid(original_hybrid):
            def __init__(self, *a, **k):
                constructed.append("hybrid")
                super().__init__(*a, **k)

        monkeypatch.setattr(vs, "HybridVectorStore", _SpyHybrid)

        outcome = _real_eval(sub_dataset, k=3)
        assert constructed, "rag 评估没有构造 HybridVectorStore (仍在走单路?)"
        assert outcome["metrics"]["case_count"] == len(SUB_CASES)

    def test_uses_hybrid_retriever_search_multi(self, monkeypatch, sub_dataset):
        """真链路断言: 底层 HybridRetriever 被真正调用过检索。"""
        from services import hybrid_retriever as hr_mod

        calls = []
        original = hr_mod.HybridRetriever._search_multi_impl

        async def _spy(self, **kwargs):
            calls.append(kwargs.get("top_k"))
            return await original(self, **kwargs)

        monkeypatch.setattr(hr_mod.HybridRetriever, "_search_multi_impl", _spy)
        _real_eval(sub_dataset, k=3)
        assert calls, "HybridRetriever 检索没被调用 (评估没走真链路)"


# ----------------------------------------------------------------------
# 3. 指标齐备
# ----------------------------------------------------------------------

class TestMetrics:
    def test_all_metric_keys_present_and_float(self, sub_dataset, monkeypatch):
        # 清 env: rerank_threshold 默认关闭必须是 None (不是 0.0)
        monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
        metrics = _real_eval(sub_dataset, k=3)["metrics"]
        for key in METRIC_KEYS:
            assert key in metrics, f"缺指标 {key}"
            assert isinstance(metrics[key], float), f"{key} 不是 float: {type(metrics[key])}"
        assert metrics["embedding_mode"] == "bow_stub"
        assert metrics["case_count"] == len(SUB_CASES)
        assert isinstance(metrics["rerank_active"], bool)
        # 阈值默认关闭: null (不是 float)
        assert metrics["rerank_threshold"] is None

    def test_empty_recall_rate_bounds(self, sub_dataset):
        metrics = _real_eval(sub_dataset, k=3)["metrics"]
        assert 0.0 <= metrics["empty_recall_rate"] <= 1.0
        assert 0.0 <= metrics["ndcg_at_k"] <= 1.0
        assert 0.0 <= metrics["recall_at_k"] <= 1.0


# ----------------------------------------------------------------------
# 4. 确定性
# ----------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_bitwise_equal(self, sub_dataset):
        """同一冻结集连跑两次: recall / mrr / ndcg 逐位相等。"""
        first = _real_eval(sub_dataset, k=3)["metrics"]
        second = _real_eval(sub_dataset, k=3)["metrics"]
        for key in ("recall_at_k", "mrr", "ndcg_at_k"):
            assert first[key] == second[key], f"{key} 两次结果不一致"

    def test_stub_embedding_is_deterministic(self):
        assert stub_embed("同一段文本") == stub_embed("同一段文本")
        assert stub_embed("同一段文本") != stub_embed("另一段文本")

    def test_bow_stub_same_text_bitwise_equal(self):
        """(§4-2) 同文本两次调用向量**逐位相等**。"""
        text = "初中数学 一元二次方程 判别式 韦达定理"
        assert stub_embed(text) == stub_embed(text)

    def test_bow_stub_more_overlap_higher_cos(self):
        """(§4-2) 词重合多的两篇 cos > 词重合少的。"""
        q = "一元二次方程 判别式 韦达定理"
        close = "初中数学 一元二次方程 判别式 韦达定理 动点问题"
        far = "小学英语 自然拼读 常见词 绘本阅读"
        assert _cos(stub_embed(q), stub_embed(close)) > _cos(stub_embed(q), stub_embed(far))


# ----------------------------------------------------------------------
# 4b. 词袋桩确有语义(词面)相关性 —— spec delta 2 §4-1
# ----------------------------------------------------------------------

class TestBowStubRelevance:
    def test_relevant_doc_scores_higher_than_irrelevant(self):
        """(§4-1) 自造 2 篇语料: query「一元二次方程怎么学」与含该词的语料 cos,
        显著大于与无关语料的 cos。"""
        from eval.rag_store import STUB_EMBED_DIM

        query_vec = stub_embed("一元二次方程 怎么学")
        relevant = stub_embed("初中数学 一元二次方程：判别式、根与系数关系 韦达定理、动点问题。")
        irrelevant = stub_embed("小学英语 自然拼读 常见词表 绘本阅读训练。")

        assert len(query_vec) == STUB_EMBED_DIM
        assert _cos(query_vec, relevant) > _cos(query_vec, irrelevant), (
            "词袋桩应让含相关词的文档得分更高 (旧 sha256 桩是纯噪声, 做不到这一点)"
        )

    def test_relevance_visible_through_real_chain(self, sub_dataset):
        """(§4-1) 端到端: 缩小语料下相关 query 能召回对应 doc (桩不再污染 dense 路)。"""
        outcome = _real_eval(sub_dataset, k=3)
        assert outcome["metrics"]["recall_at_k"] > 0.0, (
            "词袋桩下真链路应有召回; 全 0 说明桩/链路装配有问题"
        )

    def test_empty_text_returns_zero_vector(self):
        assert stub_embed("") == [0.0] * len(stub_embed("x"))
        assert stub_embed("   ") == [0.0] * len(stub_embed("x"))


# ----------------------------------------------------------------------
# 4c. 阈值默认关闭 (spec delta 2 §4-4/5/6)
# ----------------------------------------------------------------------

class TestRerankThresholdDefaultOff:
    def test_embedding_mode_is_bow_stub(self, sub_dataset):
        """(§4-3) embedding_mode 必须是 "bow_stub", 不是 "stub"。"""
        metrics = _real_eval(sub_dataset, k=3)["metrics"]
        assert metrics["embedding_mode"] == "bow_stub"

    def test_default_threshold_is_none(self, monkeypatch):
        """(§4-4) 未设 env 时 min_rerank_score is None。"""
        from services import hybrid_retriever as hr_mod

        monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
        assert hr_mod.min_rerank_score_default() is None

    def test_negative_scores_kept_when_threshold_off(self, patched_embedding_env):
        """(§4-4) 阈值默认关闭时, 负分候选**不因分为负被丢弃**。

        构造负分精排替身 (所有候选精排分 < 0) → 结果仍须保留。
        """
        from services.hybrid_retriever import HybridRetriever
        from services.knowledge_service import KnowledgeService
        from services import hybrid_retriever as hr_mod

        svc = KnowledgeService(db_path="sqlite:///:memory:")
        svc.db.add_document(content="AAAA 初中数学 一元二次方程 韦达定理", category="c", keywords=[])
        svc.db.add_document(content="BBBB 小学英语 自然拼读", category="c", keywords=[])
        _run(svc.initialize())

        class _AllNegativeReranker:
            def available(self):
                return True

            def rerank(self, _q, docs, top_k=None):
                out = []
                for doc in docs:
                    item = dict(doc)
                    item["rerank_score"] = -5.0  # 全部负分
                    out.append(item)
                return out[:top_k] if top_k is not None else out

        fake = _AllNegativeReranker()
        hr_mod.reranker.available, old_avail = fake.available, hr_mod.reranker.available
        hr_mod.reranker.rerank, old_rank = fake.rerank, hr_mod.reranker.rerank
        try:
            retriever = HybridRetriever(svc, candidates=20, rerank=True, min_rerank_score=None)
            results = _run(retriever.search("韦达定理", top_k=5))
            assert results, "阈值关闭时负分候选不应被丢弃"
            assert all(d["score"] < 0 for d in results)
        finally:
            hr_mod.reranker.available = old_avail
            hr_mod.reranker.rerank = old_rank

    def test_explicit_threshold_still_applies(self, patched_embedding_env):
        """(§4-5) 显式设阈值时机制仍生效: 负分用例被过滤 / 正分用例保留。"""
        from services.hybrid_retriever import HybridRetriever
        from services.knowledge_service import KnowledgeService
        from services import hybrid_retriever as hr_mod

        class _MixedReranker:
            def available(self):
                return True

            def rerank(self, _q, docs, top_k=None):
                out = []
                for doc in docs:
                    item = dict(doc)
                    item["rerank_score"] = 5.0 if doc["content"].startswith("AAAA") else -3.0
                    out.append(item)
                out.sort(key=lambda d: d["rerank_score"], reverse=True)
                return out[:top_k] if top_k is not None else out

        def _env():
            svc = KnowledgeService(db_path="sqlite:///:memory:")
            svc.db.add_document(content="AAAA 初中数学 韦达定理", category="c", keywords=[])
            svc.db.add_document(content="BBBB 小学英语 自然拼读", category="c", keywords=[])
            _run(svc.initialize())
            return svc

        fake = _MixedReranker()
        hr_mod.reranker.available, old_avail = fake.available, hr_mod.reranker.available
        hr_mod.reranker.rerank, old_rank = fake.rerank, hr_mod.reranker.rerank
        try:
            # 显式 0.0: 负分被丢
            strict = HybridRetriever(_env(), candidates=20, rerank=True, min_rerank_score=0.0)
            kept = _run(strict.search("韦达定理", top_k=5))
            assert kept and all(d["score"] >= 0.0 for d in kept)
            assert not any(d["content"].startswith("BBBB") for d in kept)

            # 显式 -99: 负分保留
            loose = HybridRetriever(_env(), candidates=20, rerank=True, min_rerank_score=-99.0)
            kept_all = _run(loose.search("韦达定理", top_k=5))
            assert len(kept_all) == 2, "显式负阈值下负分候选应保留"
        finally:
            hr_mod.reranker.available = old_avail
            hr_mod.reranker.rerank = old_rank

    def test_report_rerank_threshold_is_null_when_off(self, monkeypatch, tmp_path, sub_dataset):
        """(§4-6) 阈值关闭时 report 的 rag.metrics.rerank_threshold 为 null。"""
        monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
        monkeypatch.setattr(
            rag_eval_module, "evaluate_rag",
            lambda data, k=3: _REAL_EVALUATE_RAG(SUB_DATASET, k),
        )
        import eval.runner as runner_module

        out = tmp_path / "null.json"
        runner_module.main(["--only", "rag", "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        metrics = report["results"]["rag"]["metrics"]
        assert "rerank_threshold" in metrics
        assert metrics["rerank_threshold"] is None, "默认关闭时报告中应为 null"
        assert metrics["embedding_mode"] == "bow_stub"


# ----------------------------------------------------------------------
# 5. BM25 对照
# ----------------------------------------------------------------------

class TestBm25Baseline:
    def test_baseline_present(self, sub_dataset):
        metrics = _real_eval(sub_dataset, k=3)["metrics"]
        assert metrics["bm25_baseline_recall_at_k"] is not None
        assert metrics["bm25_baseline_mrr"] is not None
        assert isinstance(metrics["bm25_baseline_recall_at_k"], float)


# ----------------------------------------------------------------------
# 6. 反逐字改写守卫
# ----------------------------------------------------------------------

class TestAntiVerbatimGuard:
    def test_longest_common_substring(self):
        assert longest_common_substring_len("abcdefghij", "xxabcdefghijyy") == 10
        assert longest_common_substring_len("", "abc") == 0

    def test_verbatim_query_rejected(self):
        """自造数据: query 抄了原文连续 12+ 字 → 判不合法。"""
        corpus = ["初中数学 一元二次方程：判别式、根与系数关系(韦达定理)、动点问题。"]
        bad = [{"id": "bad_001", "query": "一元二次方程：判别式、根"}]
        violations = guard_cases(bad, corpus)
        assert len(violations) == 1
        assert violations[0]["max_common"] >= MIN_COMMON_SUBSTRING
        assert max_common_substring_len(bad[0]["query"], corpus) >= MIN_COMMON_SUBSTRING

    def test_paraphrased_query_passes(self):
        corpus = ["初中数学 一元二次方程：判别式、根与系数关系(韦达定理)、动点问题。"]
        good = [{"id": "good_001", "query": "二次方程这类题该怎么下笔"}]
        assert guard_cases(good, corpus) == []

    def test_evaluate_rag_raises_on_verbatim_query(self):
        """evaluate_rag 内部守卫: 抄原文 → 抛错 (门禁红)。"""
        dataset = {
            "version": "2.0.0", "k": 3,
            "corpus": SUB_CORPUS,
            "cases": [{
                "id": "verbatim_001",
                "query": "小学阅读启蒙课：以经典绘本为载体",
                "expected_doc_ids": [102],
                "note": "抄原文", "source_quote": "q", "label_source": "human_reviewed",
            }],
        }
        with pytest.raises(ValueError, match="反逐字改写"):
            _real_eval(dataset, k=3)

    def test_evaluate_rag_raises_on_dangling_expected(self):
        dataset = {
            "version": "2.0.0", "k": 3, "corpus": SUB_CORPUS,
            "cases": [{
                "id": "dangling_001", "query": "二次方程那类题目怎么做",
                "expected_doc_ids": [9999],
                "note": "悬空", "source_quote": "q", "label_source": "human_reviewed",
            }],
        }
        with pytest.raises(ValueError, match="悬空"):
            _real_eval(dataset, k=3)

    def test_real_rag_json_passes_guard(self):
        """磁盘上的真 rag.json 必须过守卫 (query 不许抄原文)。"""
        data = json.loads((PROJECT_ROOT / "eval" / "cases" / "rag.json").read_text(encoding="utf-8"))
        corpus_texts = [d["content"] for d in data["corpus"]]
        assert guard_cases(data["cases"], corpus_texts) == []


# ----------------------------------------------------------------------
# 7. 期望对齐 + 规模
# ----------------------------------------------------------------------

class TestRagCaseSetIntegrity:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        return json.loads((PROJECT_ROOT / "eval" / "cases" / "rag.json").read_text(encoding="utf-8"))

    def test_version_is_v2(self, data):
        assert data["version"] == "2.0.0"
        assert data["k"] == 3

    def test_corpus_has_at_least_30_docs(self, data):
        assert len(data["corpus"]) >= 30

    def test_cases_has_at_least_20(self, data):
        assert len(data["cases"]) >= 20

    def test_every_expected_doc_id_exists_in_corpus(self, data):
        corpus_ids = {int(d["doc_id"]) for d in data["corpus"]}
        for case in data["cases"]:
            assert set(case["expected_doc_ids"]).issubset(corpus_ids), (
                f"{case['id']} 期望悬空: {set(case['expected_doc_ids']) - corpus_ids}"
            )

    def test_every_case_has_source_quote_and_label_source(self, data):
        for case in data["cases"]:
            assert case.get("source_quote"), f"{case['id']} 缺 source_quote"
            assert case.get("label_source") in {"from_real_chunk", "human_reviewed"}

    def test_corpus_entries_have_required_fields(self, data):
        required = {"doc_id", "content", "keywords", "subject", "grade_level", "source_file"}
        for doc in data["corpus"]:
            assert required.issubset(doc), f"doc {doc.get('doc_id')} 缺 {required - set(doc)}"

    def test_corpus_doc_ids_unique(self, data):
        ids = [int(d["doc_id"]) for d in data["corpus"]]
        assert len(ids) == len(set(ids))

    def test_expected_docs_are_reachable(self, data):
        """ground truth 可达性: 期望 doc 至少能用自身内容被 BM25 检到。"""
        problems = rag_eval_module.check_answerable(data["corpus"], data["cases"])
        assert problems == [], f"无解 case: {problems}"


# ----------------------------------------------------------------------
# 8. 门禁负例
# ----------------------------------------------------------------------

class TestGateNegative:
    def test_all_expected_ids_bogus_makes_main_return_1(self, monkeypatch, tmp_path):
        """期望 doc_id 全改成「存在但永远检索不到」→ gates.check 判 False 且 main 返回 1。

        (悬空 id 会先被 evaluate_rag 的守卫拦下抛错, 那是另一条路径;
        这里专门测「跑得完但指标全 0」时门禁能不能真的拦住。)
        """
        reachable_bogus = {
            "version": "2.0.0", "k": 3,
            "corpus": SUB_CORPUS + [{
                "doc_id": 90001,
                "content": "与所有 query 都无关的一段占位文本，用于制造必然漏召回。",
                "keywords": [], "subject": None, "grade_level": None,
                "doc_type": None, "category": "general", "source_file": None,
            }],
            "cases": [{**c, "expected_doc_ids": [90001]} for c in SUB_CASES],
        }
        monkeypatch.setattr(
            rag_eval_module, "evaluate_rag",
            lambda data, k=3: _REAL_EVALUATE_RAG(reachable_bogus, k),
        )
        import eval.runner as runner_module
        from eval.gates import load_gates

        out = tmp_path / "neg.json"
        code = runner_module.main(["--only", "rag", "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        check = report["gates"]["check"]
        recall = report["results"]["rag"]["metrics"]["recall_at_k"]
        assert check["passed"] is False
        assert any(f["gate"].startswith("rag_") for f in check["failures"])
        # 期望全部指向"答不出问题"的诱饵文档 → 召回必然远低于门禁线
        assert recall < load_gates()["rag_min_recall_at_3"], (
            f"诱饵期望下召回应低于门禁线, 实际 {recall}"
        )
        assert code == 1, "门禁失败必须 exit 1"

    def test_empty_recall_rate_gate_is_checked(self):
        """rag_max_empty_recall_rate 是上限型门禁: 超限判失败。"""
        from eval.gates import check_gates, load_gates

        gates = load_gates()
        assert "rag_max_empty_recall_rate" in gates
        report = {
            "results": {
                "rag": {"metrics": {
                    "recall_at_k": 0.9,
                    "empty_recall_rate": gates["rag_max_empty_recall_rate"] + 0.01,
                }},
                "routing": {"metrics": {"accuracy": 0.9}},
                "slot": {"metrics": {"accuracy": 0.9}},
                "tool": {"metrics": {"accuracy": 0.9}},
                "handoff": {"metrics": {"accuracy": 0.9}},
                "safety": {"metrics": {"accuracy": 1.0}},
            },
            "metrics": {"latency_p95_ms": 1.0, "tokens": 1, "max_steps": 1},
        }
        check = check_gates(report, gates)
        assert check["passed"] is False
        assert any(f["gate"] == "rag_max_empty_recall_rate" for f in check["failures"])


# ----------------------------------------------------------------------
# 9. 门禁正例
# ----------------------------------------------------------------------

class TestGatePositive:
    def test_normal_frozen_set_passes_and_reports_new_metrics(
        self, monkeypatch, tmp_path, sub_dataset
    ):
        """rag-only 模式: rag 门禁全过 + 非 rag 门禁标 skipped + exit 0 (partial)。

        注意 **不能**断言 `code == 0` 是因为「全部门禁通过」——只跑了一类,
        另外五类没数据。正确的是: 已跑的这个类别 (rag) 没有任何 failed,
        未跑的五类显式为 `skipped`, 且 exit code 只由已跑类别决定。
        """
        monkeypatch.setattr(
            rag_eval_module, "evaluate_rag",
            lambda data, k=3: _real_eval(sub_dataset, k),
        )
        # 缩小语料上 3 条 case 全命中 → recall 1.0, 空召回 0 → 两个 rag 门禁都过
        import eval.runner as runner_module

        out = tmp_path / "pos.json"
        code = runner_module.main(["--only", "rag", "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))

        # (1) rag 段 5 个指标键齐备且为 float
        rag = report["results"]["rag"]["metrics"]
        for key in METRIC_KEYS:
            assert key in rag, f"缺指标 {key}"
            assert isinstance(rag[key], float), f"{key} 不是 float"
        assert rag["empty_recall_rate"] == 0.0
        assert rag["recall_at_k"] == 1.0

        # (2) 所有 rag_* 门禁没有一个 failed
        check = report["gates"]["check"]
        failed_gates = {f["gate"] for f in check["failures"]}
        assert not {g for g in failed_gates if g.startswith("rag_")}, (
            f"rag 门禁不该失败: {check['failures']}"
        )
        statuses = check["results"]
        for gate in ("rag_min_recall_at_3", "rag_max_empty_recall_rate"):
            assert statuses[gate] == "passed"

        # (3) 非 rag 门禁项在本模式下是 skipped (而非 failed)
        for gate in ("routing_min_accuracy", "slot_min_accuracy", "tool_min_success",
                     "handoff_min_accuracy", "safety_min_pass_rate"):
            assert statuses[gate] == "skipped", f"{gate} 应为 skipped: {statuses[gate]}"
            assert gate not in failed_gates

        # 子集模式整体结论: partial + exit 0
        assert check["status"] == "partial"
        assert check["partial"] is True
        assert check["passed"] is True
        assert code == 0, check["failures"]

    def test_partial_message_printed_on_stdout(self, monkeypatch, tmp_path, capsys, sub_dataset):
        """stdout 必须明确打印 PARTIAL: 仅校验 N 类(...),未跑 M 类(...)。"""
        monkeypatch.setattr(
            rag_eval_module, "evaluate_rag",
            lambda data, k=3: _real_eval(sub_dataset, k),
        )
        import eval.runner as runner_module

        runner_module.main(["--only", "rag", "--out", str(tmp_path / "p.json")])
        out = capsys.readouterr().out
        assert "PARTIAL: 仅校验 1 类(rag)" in out
        assert "未跑 5 类" in out
        for kind in ("routing", "slot", "tool", "handoff", "safety"):
            assert kind in out


# ----------------------------------------------------------------------
# 10. 全量模式等价性 (spec delta §3 新增: 证明 partial 语义没削弱发布门禁)
# ----------------------------------------------------------------------

class TestFullModeEquivalence:
    """不带 --only 时行为必须与改造前一致: 9 项门禁全查, 任一不达标 → exit 1。"""

    @staticmethod
    def _full_results(rag_recall: float = 1.0, safety_accuracy: float = 1.0):
        """六类齐全的 results (rag 过 + safety 可调)。"""
        return {
            "routing": {"metrics": {"accuracy": 0.9}},
            "rag": {"metrics": {"recall_at_k": rag_recall, "empty_recall_rate": 0.0}},
            "slot": {"metrics": {"accuracy": 0.9}},
            "tool": {"metrics": {"accuracy": 0.9}},
            "handoff": {"metrics": {"accuracy": 0.9}},
            "safety": {"metrics": {"accuracy": safety_accuracy}},
        }

    def test_full_mode_all_gates_checked_no_skips(self):
        """六类都跑 → 全部门禁照查, 一个 skipped 都没有。"""
        from eval.gates import check_gates, load_gates

        report = {
            "results": self._full_results(),
            "metrics": {"latency_p95_ms": 1.0, "tokens": 1, "max_steps": 1},
            "meta": {"requested_kinds": _requested_all_kinds()},
        }
        check = check_gates(report, load_gates())
        assert check["skipped"] == []
        assert check["partial"] is False
        assert check["status"] == "passed"
        # 7 项类别门禁 (routing / rag×2 / slot / tool / handoff / safety)
        # + 3 项全局门禁 (P95 / token / steps) = 10 项全查
        # (改造前是 9 项, 本轮新增 rag_max_empty_recall_rate)
        assert check["checked"] == 10
        assert set(check["results"].values()) == {"passed"}

    def test_rag_pass_safety_fail_returns_1(self):
        """rag 过 + safety 不过 → gates.check 判 False 且 main 返回 1。

        这是 partial 改动**没有**削弱发布门禁的证据: 全量模式下不达标照样拦住。
        """
        from eval.gates import check_gates, load_gates

        gates = load_gates()
        report = {
            "results": self._full_results(rag_recall=1.0, safety_accuracy=0.5),
            "metrics": {"latency_p95_ms": 1.0, "tokens": 1, "max_steps": 1},
        }
        check = check_gates(report, gates)
        assert check["passed"] is False
        assert check["status"] == "failed"
        assert check["partial"] is False
        assert any(f["gate"] == "safety_min_pass_rate" for f in check["failures"])
        assert check["results"]["rag_min_recall_at_3"] == "passed"
        assert check["results"]["safety_min_pass_rate"] == "failed"

    def test_rag_fail_in_full_mode_also_returns_1(self):
        """全量模式下 rag 不达标同样 exit 1 (改造前后一致)。"""
        from eval.gates import check_gates, load_gates

        gates = load_gates()
        report = {
            "results": self._full_results(rag_recall=0.10),
            "metrics": {"latency_p95_ms": 1.0, "tokens": 1, "max_steps": 1},
        }
        check = check_gates(report, gates)
        assert check["passed"] is False
        assert any(f["gate"] == "rag_min_recall_at_3" for f in check["failures"])

    def test_main_returns_1_in_full_mode_when_safety_fails(self, monkeypatch, tmp_path):
        """端到端: 全量模式 (不带 --only) 下 safety 不过 → `main` 返回 1。"""
        import eval.runner as runner_module
        from eval.runner import EDDRunner, _summarize

        def _fake(kind, ok: bool = True, accuracy: float = 0.9, empty_rate: float = 0.0):
            def _method(self):
                # accuracy 必须由「通过数/总数」体现 (check_gates 读的是 metrics.accuracy)
                return _summarize(kind, [{
                    "id": f"{kind}_x", "ok": ok, "latency_ms": 1.0,
                    "tokens": 0, "steps": 0, "expected": {}, "actual": {},
                }], extra_metrics=(
                    {"recall_at_k": accuracy, "empty_recall_rate": empty_rate}
                    if kind == "rag" else None
                ))
            return _method

        for kind in ("routing", "rag", "slot", "tool", "handoff"):
            monkeypatch.setattr(EDDRunner, f"_evaluate_{kind}", _fake(kind))
        # safety 全灭 → metrics.accuracy == 0.0 < 1.0 门禁阈值
        monkeypatch.setattr(
            EDDRunner, "_evaluate_safety", _fake("safety", ok=False, accuracy=0.0)
        )
        # 全量模式会创建 LLM → 打桩掉, 不依赖任何凭据
        monkeypatch.setattr(
            "config.model_provider.create_chat_model", lambda **_k: object(), raising=False,
        )

        out = tmp_path / "full.json"
        code = runner_module.main(["--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        check = report["gates"]["check"]
        assert check["status"] == "failed"
        assert check["partial"] is False
        assert check["skipped"] == []
        assert any(f["gate"] == "safety_min_pass_rate" for f in check["failures"])
        assert code == 1, "全量模式下任一门禁不达标必须 exit 1"


# ----------------------------------------------------------------------
# 补充: 指标函数本身
# ----------------------------------------------------------------------

class TestMetricFunctions:
    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], 3) == pytest.approx(1.0)

    def test_ndcg_miss_is_zero(self):
        assert ndcg_at_k([9, 8, 7], [1], 3) == 0.0

    def test_ndcg_lower_when_hit_is_later(self):
        assert ndcg_at_k([9, 1], [1], 3) < ndcg_at_k([1, 9], [1], 3)

    def test_dcg_and_ideal(self):
        assert ideal_dcg_at_k([1, 2], 2) == pytest.approx(1.0 + 1.0 / 1.584962500721156)
        assert dcg_at_k([1], [1], 3) == pytest.approx(1.0)

    def test_empty_recall_rate(self):
        assert empty_recall_rate([]) == 0.0
        assert empty_recall_rate([{"recall": 0.0}, {"recall": 1.0}]) == 0.5
        assert empty_recall_rate([{"recall": 1.0}]) == 0.0
