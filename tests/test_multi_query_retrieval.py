"""多查询检索(search_multi + RRF 融合 + 降级聚合 + 接线)单元测试。

覆盖:
1. search_multi 多查询: 同一 doc 只出现一次、rank 连续、结果非空
2. Bm25Index.build 调用次数 == 1(多查询不重复建索引)
3. 单条 query 的 dense 抛错 → 其它 query 的结果仍在
4. 全部 query 皆空 → [], 不抛
5. search(q) 与 search_multi([q]) 结果一致(薄包装等价性)
6. degraded 聚合: 只有一条 query 的 dense 失败 → bm25_only
7. KnowledgeService.search_multi 去重融合正确
8. KnowledgeRetriever.search_knowledge 在 mode=off 时只对原始 query 检索一次

Embedding 用确定性 fake(sha256→16 维), 不依赖远程 API。
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _fake_embed(text: str) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input: text_embedding / knowledge_service / hybrid_retriever 三处。"""
    from services import text_embedding, knowledge_service, hybrid_retriever

    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)
    monkeypatch.setattr(knowledge_service, "embed_input", _fake_embed)
    monkeypatch.setattr(hybrid_retriever, "embed_input", _fake_embed)


@pytest.fixture
def rag_spans():
    """捕获 search_multi 自己发出的 rag_search span(其 attrs 挂在内层 span 上)。"""
    import logging

    captured: List[Dict] = []
    logger_obs = logging.getLogger("obs.trace")
    old_level = logger_obs.level

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(getattr(record, "trace", None))

    handler = _CaptureHandler()
    logger_obs.addHandler(handler)
    logger_obs.setLevel(logging.INFO)  # 默认 root 是 WARNING, 不放开收不到 INFO trace
    try:
        yield captured
    finally:
        logger_obs.removeHandler(handler)
        logger_obs.setLevel(old_level)


def _seed_corpus(repo) -> List[int]:
    """种子 4 条: 学科/年级/类别组合, 专有名词突出便于 BM25 验证。"""
    return [
        repo.add_document(
            content="高中数学 12 周提分方案: 函数、几何、数列、概率四大模块, 含韦达定理专题。",
            category="course_intro", keywords=["高中", "数学", "高考", "韦达定理"],
            subject="数学", grade_level="高中", doc_type="course_intro",
        ),
        repo.add_document(
            content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
            category="course_intro", keywords=["小学", "阅读", "绘本"],
            subject="语文", grade_level="小学", doc_type="course_intro",
        ),
        repo.add_document(
            content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
            category="subject_method", keywords=["初中", "数学", "一元二次方程", "韦达定理"],
            subject="数学", grade_level="初中", doc_type="subject_method",
        ),
        repo.add_document(
            content="如何帮孩子选家教老师: 看学科匹配、年级段、教学风格, 建议先试讲。",
            category="parent_guide", keywords=["家长", "老师", "试讲"],
            subject=None, grade_level=None, doc_type="parent_guide",
        ),
    ]


@pytest.fixture
def hybrid_env():
    """内存库 + 种子数据 + 已初始化的 HybridRetriever(rerank 关)。"""
    from services.knowledge_service import KnowledgeService
    from services.hybrid_retriever import HybridRetriever

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    _seed_corpus(svc.db)
    _run(svc.initialize())

    retriever = HybridRetriever(svc, rrf_k=60, candidates=20, rerank=False)
    return {"svc": svc, "retriever": retriever}


@pytest.fixture(autouse=True)
def _reset_transform_settings(monkeypatch):
    """默认档回 rule, 避免真实 .env 泄漏进用例。"""
    monkeypatch.setattr(settings, "query_transform_mode", "rule")
    monkeypatch.setattr(settings, "query_transform_max_variants", 3)


# ----------------------------------------------------------------------
# HybridRetriever.search_multi
# ----------------------------------------------------------------------

class TestHybridSearchMulti:
    def test_multi_query_dedups_and_ranks_continuously(self, hybrid_env):
        """1. 同一 doc 只出现一次、rank 连续、结果非空。"""
        retriever = hybrid_env["retriever"]
        results = _run(retriever.search_multi(
            ["韦达定理 初中数学", "一元二次方程 判别式"], top_k=5,
        ))
        assert results, "多查询检索不应为空"
        ids = [d["id"] for d in results]
        assert len(ids) == len(set(ids)), f"同一 doc 出现了多次: {ids}"
        assert [d["rank"] for d in results] == list(range(1, len(results) + 1))

    def test_bm25_index_built_once_for_multiple_queries(self, hybrid_env, monkeypatch):
        """2. Bm25Index.build 调用次数 == 1(多查询不重复建索引)。"""
        from services import hybrid_retriever as hr_mod

        build_calls = {"n": 0}
        real_build = hr_mod.Bm25Index.build

        def _counting_build(self, docs):
            build_calls["n"] += 1
            return real_build(self, docs)

        monkeypatch.setattr(hr_mod.Bm25Index, "build", _counting_build)

        retriever = hybrid_env["retriever"]
        results = _run(retriever.search_multi(
            ["韦达定理", "一元二次方程", "初中数学"], top_k=5,
        ))
        assert results
        assert build_calls["n"] == 1, f"BM25 索引被重建了 {build_calls['n']} 次"

    def test_one_query_dense_failure_does_not_break_others(self, hybrid_env, monkeypatch):
        """3. 某条 query 的 dense 抛错 → 其它 query 的结果仍在。"""
        from services import hybrid_retriever as hr_mod

        real_dense = hr_mod.HybridRetriever._dense_search
        seen: List[str] = []

        def _flaky(self, query, candidates):
            seen.append(query)
            if len(seen) == 1:
                raise RuntimeError("模拟第一条 query 的 embedding 失败")
            return real_dense(self, query, candidates)

        monkeypatch.setattr(hr_mod.HybridRetriever, "_dense_search", _flaky)

        retriever = hybrid_env["retriever"]
        results = _run(retriever.search_multi(
            ["韦达定理 初中数学", "一元二次方程 判别式", "绘本 阅读"], top_k=5,
        ))
        assert results, "单条 query 失败不应连累整轮检索"
        assert len(seen) == 3, "每条 query 都应各跑一次 dense"

    def test_all_queries_empty_returns_empty_list(self, hybrid_env, monkeypatch):
        """4. 全部 query 皆空(两路都无命中)→ [], 不抛。

        FAISS 内积检索对任何 query 都会返回近邻, 所以「全空」要在 dense 路也断掉
        (模拟 embedding 不可用)+ BM25 无词命中, 才是真实的两路皆空。
        """
        from services import hybrid_retriever as hr_mod

        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )
        retriever = hybrid_env["retriever"]
        results = _run(retriever.search_multi(
            ["量子纠缠", "弦论超导", "虫洞白洞"], top_k=3,
        ))
        assert results == []

    def test_search_is_thin_wrapper_of_search_multi(self, hybrid_env):
        """5. search(q) 与 search_multi([q]) 结果一致(薄包装等价性)。"""
        retriever = hybrid_env["retriever"]
        single = _run(retriever.search("韦达定理 初中数学", top_k=3))
        multi = _run(retriever.search_multi(["韦达定理 初中数学"], top_k=3))

        assert single == multi
        assert single, "等价性断言需要非空结果才有意义"

    def test_degraded_aggregation_bm25_only(self, hybrid_env, monkeypatch, rag_spans):
        """6. 只有一条 query 的 dense 失败 → 聚合 degraded == 'bm25_only'。"""
        from services import hybrid_retriever as hr_mod

        real_dense = hr_mod.HybridRetriever._dense_search
        seen: List[str] = []

        def _flaky(self, query, candidates):
            seen.append(query)
            if len(seen) == 1:
                raise RuntimeError("dense down")
            return real_dense(self, query, candidates)

        monkeypatch.setattr(hr_mod.HybridRetriever, "_dense_search", _flaky)

        retriever = hybrid_env["retriever"]
        results = _run(retriever.search_multi(
            ["韦达定理 初中数学", "一元二次方程 判别式"], top_k=5,
        ))
        assert results
        spans = [s for s in rag_spans if s and s["name"] == "rag_search"]
        assert len(spans) == 1
        assert spans[0]["attrs"]["degraded"] == "bm25_only"
        assert spans[0]["attrs"]["variants"] == 2
        assert spans[0]["attrs"]["degraded_per_query"] == ["bm25_only", "none"]

    def test_degraded_aggregation_all_empty(self, hybrid_env, monkeypatch, rag_spans):
        """6b. 两路全空但有异常 → 'empty'; 无异常 → 'no_hit'。"""
        from services import hybrid_retriever as hr_mod

        retriever = hybrid_env["retriever"]

        # 有异常(dense 挂掉 + BM25 无词命中)→ empty
        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )
        _run(retriever.search_multi(["量子纠缠", "弦论超导"], top_k=3))

        # 无异常: 两条 query 都只命中词典外的词 → BM25 空; dense 被摘掉 → no_hit
        # 这里直接断言聚合函数本身, 避免为「无异常且两路皆空」构造不现实的桩。
        assert hr_mod.HybridRetriever._aggregate_degraded(
            ["no_hit", "no_hit"], any_exception=False,
        ) == "no_hit"
        assert hr_mod.HybridRetriever._aggregate_degraded(
            ["no_hit", "no_hit"], any_exception=True,
        ) == "empty"
        assert hr_mod.HybridRetriever._aggregate_degraded(
            ["dense_only", "none"], any_exception=False,
        ) == "dense_only"
        assert hr_mod.HybridRetriever._aggregate_degraded(
            ["none", "none"], any_exception=False,
        ) == "none"
        assert hr_mod.HybridRetriever._aggregate_degraded(
            ["bm25_only", "dense_only"], any_exception=True,
        ) == "bm25_only"

        spans = [s for s in rag_spans if s and s["name"] == "rag_search"]
        assert len(spans) == 1
        assert spans[0]["attrs"]["degraded"] == "empty"

    def test_empty_queries_list_does_not_raise(self, hybrid_env):
        """4b. 空 queries 列表 → 按单条空串处理, 不抛异常。"""
        retriever = hybrid_env["retriever"]
        assert isinstance(_run(retriever.search_multi([], top_k=3)), list)

    def test_multi_query_can_recall_more_than_single(self, hybrid_env):
        """多路召回是增量: 变体让原本漏掉的文档也能进候选(不降召回)。"""
        retriever = hybrid_env["retriever"]
        single = _run(retriever.search("韦达定理", top_k=2))
        multi = _run(retriever.search_multi(
            ["韦达定理", "高中 数学 提分方案"], top_k=2,
        ))
        assert {d["id"] for d in single} <= {d["id"] for d in multi}


# ----------------------------------------------------------------------
# KnowledgeService.search_multi(dense 路径)
# ----------------------------------------------------------------------

class TestKnowledgeServiceSearchMulti:
    def test_fuse_dedups_and_ranks_continuously(self, hybrid_env):
        """7. KnowledgeService.search_multi 去重融合正确、rank 连续、score 为融合分。"""
        svc = hybrid_env["svc"]
        results = _run(svc.search_multi(
            ["韦达定理", "一元二次方程 判别式", "初中数学"], top_k=4,
        ))
        assert results, "多查询 dense 检索不应为空"
        ids = [d["id"] for d in results]
        assert len(ids) == len(set(ids)), f"同一 doc 出现了多次: {ids}"
        assert [d["rank"] for d in results] == list(range(1, len(results) + 1))
        # 融合分是 1/(k+rank) 量级的小正数(跨 query 唯一可比的分数量纲)
        assert all(0.0 < d["score"] < 1.0 for d in results)

    def test_single_query_matches_search(self, hybrid_env):
        """7b. 单条 query 时 search_multi([q]) 与 search(q) 逐字一致。"""
        svc = hybrid_env["svc"]
        single = _run(svc.search("韦达定理 初中数学", top_k=3))
        multi = _run(svc.search_multi(["韦达定理 初中数学"], top_k=3))
        assert single == multi
        assert single

    def test_fuse_ranks_intersection_docs_first(self, hybrid_env):
        """7c. 被两条 query 同时命中的文档, 融合分应不低于只被一条命中的文档。

        top_k 取 2 保证两条 query 的候选集不完全重合(全集重合时交集==并集, 断言无意义)。
        """
        svc = hybrid_env["svc"]
        queries = ["韦达定理 初中数学", "一元二次方程 判别式"]
        fused = _run(svc.search_multi(queries, top_k=2))
        assert fused

        per_query = [
            {d["id"] for d in _run(svc.search(q, top_k=2))} for q in queries
        ]
        both = per_query[0] & per_query[1]
        only_one = (per_query[0] | per_query[1]) - both
        assert both and only_one, f"候选集构造无意义: {per_query}"

        scores = {d["id"]: d["score"] for d in fused}
        assert min(scores[i] for i in both if i in scores) >= max(
            scores[i] for i in only_one if i in scores
        )

    def test_all_queries_empty_returns_empty_list(self, hybrid_env):
        """4c. dense 路径全部 query 无命中 → []。"""
        svc = hybrid_env["svc"]
        svc.index = None
        svc.initialized = False
        assert _run(svc.search_multi(["韦达定理", "一元二次方程"], top_k=3)) == []


# ----------------------------------------------------------------------
# KnowledgeRetriever 接线
# ----------------------------------------------------------------------

def _make_retriever(mock_multi, mode: str = "hybrid"):
    """构造 KnowledgeRetriever, 用 mock 顶替底层检索(隔离真实检索 / LLM)。"""
    from agents.consultant.knowledge_retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    retriever.retrieval_mode = mode
    retriever.kb_initialized = True  # 跳过真实初始化

    service = MagicMock()
    service.search_multi = mock_multi
    retriever.knowledge_service = service

    hybrid = MagicMock()
    hybrid.search_multi = mock_multi
    retriever.hybrid_retriever = hybrid
    return retriever


class TestKnowledgeRetrieverWiring:
    def test_off_mode_searches_only_original_query(self, monkeypatch):
        """8. mode=off 时只对原始 query 检索一次(可 patch 底层计数)。"""
        monkeypatch.setattr(settings, "query_transform_mode", "off")
        mock = AsyncMock(return_value=[{"content": "doc"}])
        retriever = _make_retriever(mock, mode="hybrid")

        _run(retriever.search_knowledge("试听多少钱", top_k=3))

        mock.assert_awaited_once_with(
            ["试听多少钱"], top_k=3, subject=None, grade_level=None,
        )

    def test_rule_mode_passes_multiple_variants(self, monkeypatch):
        """8b. mode=rule 时把术语归一 + 指代补全的变体一起送进底层检索, [0] 仍是原串。"""
        monkeypatch.setattr(settings, "query_transform_mode", "rule")
        monkeypatch.setattr(settings, "query_transform_max_variants", 3)
        mock = AsyncMock(return_value=[{"content": "doc"}])
        retriever = _make_retriever(mock, mode="hybrid")

        context = "用户: 我家孩子初二,数学不好"
        _run(retriever.search_knowledge(
            "试听这个怎么样", top_k=3, query_context=context,
        ))

        passed = mock.await_args.args[0]
        effective = f"{context}\n试听这个怎么样"
        assert passed[0] == effective, "原串(拼接后的 effective_query)必须在 [0]"
        assert len(passed) == 3, f"应送 3 条检索串: {passed}"
        assert f"{context}\n试讲这个怎么样" in passed, "术语归一变体应在列表里"
        assert f"初中数学：{effective}" in passed, "指代补全变体应在列表里"

    def test_dense_path_also_uses_search_multi(self, monkeypatch):
        """8c. retrieval_mode=dense 同样走 search_multi(两路分流不变)。"""
        monkeypatch.setattr(settings, "query_transform_mode", "off")
        mock = AsyncMock(return_value=[])
        retriever = _make_retriever(mock, mode="dense")

        _run(retriever.search_knowledge("计算薄弱", top_k=3))

        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["计算薄弱"], top_k=3, subject=None, grade_level=None,
        )

    def test_query_context_still_spliced_into_first_query(self, monkeypatch):
        """向后兼容: query_context 硬拼接保留, [0] == 拼接后的 effective_query。"""
        monkeypatch.setattr(settings, "query_transform_mode", "off")
        mock = AsyncMock(return_value=[])
        retriever = _make_retriever(mock, mode="hybrid")

        _run(retriever.search_knowledge(
            "计算薄弱", top_k=3, query_context="用户: 初二,数学不行",
        ))

        mock.assert_awaited_once_with(
            ["用户: 初二,数学不行\n计算薄弱"], top_k=3,
            subject=None, grade_level=None,
        )

    def test_search_failure_still_degrades_to_empty(self, monkeypatch):
        """失败降级链一字不改: 底层抛错 → 空召回, 不冒泡。"""
        monkeypatch.setattr(settings, "query_transform_mode", "rule")
        mock = AsyncMock(side_effect=RuntimeError("检索炸了"))
        retriever = _make_retriever(mock, mode="hybrid")

        assert _run(retriever.search_knowledge("试听多少钱", top_k=3)) == []
