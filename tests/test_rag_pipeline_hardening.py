"""RAG 检索 pipeline 加固(P0 三项 + 三级降级)单元测试。

覆盖:
1. 默认 RETRIEVAL_MODE == hybrid
2. ChunkerSettings 默认 500/50 且 == config/components.yaml 的 splitter 声明(防漂移)
3. dense 路抛异常 → search() 不抛, 仍返回 BM25 命中
4. dense + BM25 同时失败/无命中 → 返回 [] 不抛
5. 精排阈值: 负分丢弃 / 正分保留
6. RRF 兜底路径(rerank=False)阈值不生效
7. 去重: 重复内容只留一条, rank 连续
8. prompt_builder 来源编号 + 空输入兜底文案原文

Embedding 用确定性 fake(sha256→16 维), 不依赖远程 API。
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _seed_corpus(repo) -> List[int]:
    """种子 3 条, 专有名词突出便于 BM25 验证。"""
    return [
        repo.add_document(
            content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
            category="subject_method", keywords=["初中", "数学", "韦达定理"],
            subject="数学", grade_level="初中", doc_type="subject_method",
        ),
        repo.add_document(
            content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
            category="course_intro", keywords=["小学", "阅读", "绘本"],
            subject="语文", grade_level="小学", doc_type="course_intro",
        ),
        repo.add_document(
            content="如何帮孩子选家教老师: 看学科匹配、年级段、教学风格。",
            category="parent_guide", keywords=["家长", "老师"],
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


# ----------------------------------------------------------------------
# 1. 默认检索模式
# ----------------------------------------------------------------------

class TestRetrievalModeDefault:
    def test_default_is_hybrid(self, monkeypatch):
        """未设 RETRIEVAL_MODE 时默认 hybrid(与 .env.example / README 声明一致)。"""
        from config.settings import AppSettings

        monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
        assert AppSettings().retrieval_mode == "hybrid"

    def test_env_override_still_wins(self, monkeypatch):
        """env 仍可显式覆盖回 dense(向后兼容)。"""
        from config.settings import AppSettings

        monkeypatch.setenv("RETRIEVAL_MODE", "dense")
        assert AppSettings().retrieval_mode == "dense"


# ----------------------------------------------------------------------
# 2. 分块默认值 + 与 components.yaml 防漂移
# ----------------------------------------------------------------------

class TestChunkerDefaults:
    def test_defaults_are_500_50(self):
        from services.text_chunker import ChunkerSettings

        s = ChunkerSettings()
        assert s.chunk_size == 500
        assert s.chunk_overlap == 50

    def test_settings_defaults_match_components_yaml(self):
        """settings 的分块默认值必须 == config/components.yaml 的 splitter 声明。

        任一处改动漏改另一处会立刻红(死配置守卫)。
        """
        import yaml

        from config.settings import AppSettings

        yaml_path = PROJECT_ROOT / "config" / "components.yaml"
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["components"]["splitter"]
        s = AppSettings()
        assert (s.chunk_size, s.chunk_overlap) == (spec["chunk_size"], spec["overlap"]), (
            f"settings({s.chunk_size}/{s.chunk_overlap}) 与 components.yaml"
            f"({spec['chunk_size']}/{spec['overlap']}) 不一致"
        )

    def test_chunker_defaults_match_settings_defaults(self, monkeypatch):
        """ChunkerSettings 默认值也必须与 settings 一致(上传路径靠 settings 传参)。"""
        from config.settings import AppSettings
        from services.text_chunker import ChunkerSettings

        monkeypatch.delenv("CHUNK_SIZE", raising=False)
        monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
        s = AppSettings()
        c = ChunkerSettings()
        assert (c.chunk_size, c.chunk_overlap) == (s.chunk_size, s.chunk_overlap)


# ----------------------------------------------------------------------
# 3./4. 三级降级
# ----------------------------------------------------------------------

class TestThreeTierDegradation:
    def test_dense_failure_falls_back_to_bm25(self, hybrid_env, monkeypatch, caplog):
        """embed_input 抛异常 → search() 不抛, 且返回 BM25 命中(非空)。"""
        from services import hybrid_retriever as hr_mod

        def _boom(_q):
            raise RuntimeError("模拟 embedding API 429")

        monkeypatch.setattr(hr_mod, "embed_input", _boom)
        retriever = hybrid_env["retriever"]

        results = _run(retriever.search("韦达定理 一元二次方程", top_k=3))
        assert results, "dense 失败后必须回退到 BM25 并返回命中"
        assert any("韦达定理" in d["content"] for d in results)

    def test_dense_failure_marks_degraded(self, hybrid_env, monkeypatch):
        """span.attrs['degraded'] 标记 bm25_only, 便于压测/排查。"""
        from services import hybrid_retriever as hr_mod
        from observability import trace_span

        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )
        retriever = hybrid_env["retriever"]

        with trace_span("test_rag_search") as span:
            results = _run(retriever._search_impl(
                query="韦达定理", top_k=3, category=None, subject=None,
                grade_level=None, doc_type=None, span=span,
            ))
        assert results
        assert span.attrs.get("degraded") == "bm25_only"

    def test_both_paths_empty_returns_empty_list(self, hybrid_env, monkeypatch):
        """dense 失败 + BM25 无命中 → 返回 [], 不抛。"""
        from services import hybrid_retriever as hr_mod

        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )
        retriever = hybrid_env["retriever"]

        # 库里完全没有的词 → BM25 无命中
        results = _run(retriever.search("量子纠缠 弦论 超导", top_k=3))
        assert results == []

    def test_both_paths_fail_returns_empty_list(self, hybrid_env, monkeypatch):
        """dense 与 BM25 同时失败 → 返回 [], 不抛。"""
        from services import hybrid_retriever as hr_mod

        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )

        class _BoomBm25:
            def build(self, _candidates):
                raise RuntimeError("bm25 down")

            def search(self, _query, _top_n):  # pragma: no cover - build 先炸
                raise RuntimeError("bm25 down")

        monkeypatch.setattr(hr_mod, "Bm25Index", _BoomBm25)
        retriever = hybrid_env["retriever"]

        results = _run(retriever.search("韦达定理", top_k=3))
        assert results == []

    def test_both_paths_failed_marks_degraded_empty(self, hybrid_env, monkeypatch):
        """两路都失败 → span.attrs['degraded'] == 'empty'。"""
        from services import hybrid_retriever as hr_mod
        from observability import trace_span

        monkeypatch.setattr(
            hr_mod, "embed_input",
            lambda _q: (_ for _ in ()).throw(RuntimeError("dense down")),
        )

        class _BoomBm25:
            def build(self, _candidates):
                raise RuntimeError("bm25 down")

            def search(self, _query, _top_n):  # pragma: no cover
                raise RuntimeError("bm25 down")

        monkeypatch.setattr(hr_mod, "Bm25Index", _BoomBm25)
        retriever = hybrid_env["retriever"]

        with trace_span("test_rag_search") as span:
            results = _run(retriever._search_impl(
                query="韦达定理", top_k=3, category=None, subject=None,
                grade_level=None, doc_type=None, span=span,
            ))
        assert results == []
        assert span.attrs.get("degraded") == "empty"

    def test_healthy_path_degraded_none(self, hybrid_env):
        """两路都正常 → degraded == 'none'。"""
        from observability import trace_span

        retriever = hybrid_env["retriever"]
        with trace_span("test_rag_search") as span:
            results = _run(retriever._search_impl(
                query="韦达定理", top_k=3, category=None, subject=None,
                grade_level=None, doc_type=None, span=span,
            ))
        assert results
        assert span.attrs.get("degraded") == "none"


# ----------------------------------------------------------------------
# 5./6. 精排阈值
# ----------------------------------------------------------------------

class _FakeReranker:
    """按 content 里的标记返回固定分数, 避免依赖真实模型。"""

    def __init__(self, scores: Dict[str, float]):
        self.scores = scores

    def available(self) -> bool:
        return True

    def rerank(self, _query, docs, top_k=None):
        out = []
        for doc in docs:
            marker = doc["content"].split()[0]
            if marker not in self.scores:
                continue
            item = dict(doc)
            item["rerank_score"] = self.scores[marker]
            out.append(item)
        out.sort(key=lambda d: d["rerank_score"], reverse=True)
        return out[:top_k] if top_k is not None else out


@pytest.fixture
def score_env(monkeypatch):
    """3 条候选, 精排分: 5.0 / -3.0 / 4.0(负分应被默认阈值丢弃)。"""
    from services.knowledge_service import KnowledgeService
    from services.hybrid_retriever import HybridRetriever
    from services import hybrid_retriever as hr_mod

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    svc.db.add_document(
        content="AAAA 初中数学 一元二次方程 韦达定理 判别式",
        category="subject_method", keywords=["数学"],
    )
    svc.db.add_document(
        content="BBBB 小学英语 自然拼读 常见词",
        category="subject_method", keywords=["英语"],
    )
    svc.db.add_document(
        content="CCCC 初中数学 根与系数关系 韦达定理 动点",
        category="subject_method", keywords=["数学"],
    )
    _run(svc.initialize())

    fake = _FakeReranker({"AAAA": 5.0, "BBBB": -3.0, "CCCC": 4.0})
    monkeypatch.setattr(hr_mod.reranker, "available", fake.available)
    monkeypatch.setattr(hr_mod.reranker, "rerank", fake.rerank)
    # 清掉 env, 让 retriever 落到「默认不设阈值(None)」这条路径上
    monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
    return HybridRetriever(svc, candidates=20, rerank=True)


class TestRerankThreshold:
    def test_default_is_no_threshold_and_keeps_negative(self, score_env):
        """**默认不设阈值**: min_rerank_score 为 None → 负分候选也保留(不被丢弃)。"""
        assert score_env.min_rerank_score is None, (
            "精排阈值默认应为 None(不设阈值); 默认 0.0 会把半数正确答案清零"
        )
        results = _run(score_env.search("韦达定理 初中数学", top_k=5))

        assert results
        markers = {d["content"].split()[0] for d in results}
        # BBBB 精排分 -3.0, 默认不过滤时必须在结果里
        assert markers == {"AAAA", "BBBB", "CCCC"}, "默认应保留负分候选"

    def test_explicit_threshold_drops_negative_scores(self, score_env):
        """**显式**设置阈值时机制仍生效: 负分候选被丢弃, 正分保留。"""
        score_env.min_rerank_score = 0.0
        results = _run(score_env.search("韦达定理 初中数学", top_k=5))

        assert results
        assert all(d["score"] >= 0.0 for d in results)
        assert not any(d["content"].startswith("BBBB") for d in results)
        assert {d["content"].split()[0] for d in results} == {"AAAA", "CCCC"}

    def test_threshold_disabled_keeps_negative(self):
        """阈值设成很低(等价关闭)时负分候选保留 → 证明阈值真的在起作用。"""
        import asyncio

        from services.knowledge_service import KnowledgeService
        from services.hybrid_retriever import HybridRetriever
        from services import hybrid_retriever as hr_mod

        svc = KnowledgeService(db_path="sqlite:///:memory:")
        svc.db.add_document(content="AAAA 韦达定理 判别式", category="c", keywords=[])
        svc.db.add_document(content="BBBB 自然拼读", category="c", keywords=[])
        asyncio.run(svc.initialize())

        fake = _FakeReranker({"AAAA": 5.0, "BBBB": -3.0})
        hr_mod.reranker.available, old_avail = fake.available, hr_mod.reranker.available
        hr_mod.reranker.rerank, old_rank = fake.rerank, hr_mod.reranker.rerank
        try:
            off = HybridRetriever(svc, candidates=20, rerank=True, min_rerank_score=-99.0)
            kept = asyncio.run(off.search("韦达定理", top_k=5))
            assert len(kept) == 2, "阈值关闭时负分候选应保留"
        finally:
            hr_mod.reranker.available = old_avail
            hr_mod.reranker.rerank = old_rank

    def test_rrf_fallback_ignores_threshold(self, hybrid_env):
        """rerank=False(走 RRF 兜底)时, 阈值不生效 → 结果条数不受阈值影响。"""
        retriever = hybrid_env["retriever"]
        retriever.min_rerank_score = 999.0  # 若误用会过滤掉全部结果
        results = _run(retriever.search("韦达定理 初中数学", top_k=3))
        assert results, "RRF 兜底路径不应应用精排阈值"
        assert all("rerank_score" not in d for d in results)


# ----------------------------------------------------------------------
# 7. 去重
# ----------------------------------------------------------------------

class TestDedup:
    def test_duplicate_content_kept_once_with_contiguous_rank(self):
        from services.hybrid_retriever import dedup_docs

        docs = [
            {"id": 1, "content": "初中数学 韦达定理", "category": "a", "score": 5.0},
            {"id": 2, "content": "初中数学韦达定理", "category": "a", "score": 4.0},
            {"id": 3, "content": "小学阅读 绘本", "category": "b", "score": 3.0},
        ]
        out = dedup_docs(docs)
        assert [d["id"] for d in out] == [1, 3], "重复内容只保留排名靠前的一条"

    def test_same_source_hash_and_chunk_index_is_same_doc(self):
        from services.hybrid_retriever import dedup_docs

        docs = [
            {"id": 1, "content": "内容 A", "source_hash": "h1", "chunk_index": 0},
            {"id": 2, "content": "内容 B 不同但同一块", "source_hash": "h1", "chunk_index": 0},
        ]
        out = dedup_docs(docs)
        assert [d["id"] for d in out] == [1]

    def test_search_result_rank_is_contiguous_after_dedup(self, score_env):
        """端到端: 去重后 rank 必须连续(1..n)。"""
        score_env.min_rerank_score = -99.0
        results = _run(score_env.search("韦达定理 初中数学", top_k=5))
        assert results
        assert [d["rank"] for d in results] == list(range(1, len(results) + 1))


# ----------------------------------------------------------------------
# 8. prompt_builder 来源引用
# ----------------------------------------------------------------------

class TestPromptBuilderCitations:
    def test_context_has_citation_markers(self):
        from agents.consultant.prompt_builder import PromptBuilder

        docs = [
            {"content": "初中数学 韦达定理", "category": "subject_method"},
            {"content": "小学阅读 绘本", "category": "course_intro"},
        ]
        text = PromptBuilder().build_knowledge_context(docs)
        assert "[资料1|subject_method]" in text
        assert "[资料2|course_intro]" in text
        assert "初中数学 韦达定理" in text

    def test_missing_category_uses_placeholder(self):
        from agents.consultant.prompt_builder import PromptBuilder

        text = PromptBuilder().build_knowledge_context([{"content": "内容 X"}])
        assert "[资料1|未知]" in text

    def test_empty_input_keeps_original_fallback_text(self):
        from agents.consultant.prompt_builder import PromptBuilder

        text = PromptBuilder().build_knowledge_context([])
        assert text == "没有找到直接相关的知识库信息，请基于教育学常识回答，并标明「仅供参考」。"

    def test_non_empty_keeps_trailing_hint_verbatim(self):
        from agents.consultant.prompt_builder import PromptBuilder

        text = PromptBuilder().build_knowledge_context(
            [{"content": "内容 X", "category": "faq"}]
        )
        assert "请基于以上信息回答用户问题。如果知识库信息不足以回答问题，请基于教育学常识补充回答。" in text


# ----------------------------------------------------------------------
# 补充: reranker max_length 可配
# ----------------------------------------------------------------------

class TestRerankerMaxLength:
    def test_default_is_512(self, monkeypatch):
        from services import reranker

        monkeypatch.delenv("RERANK_MAX_LENGTH", raising=False)
        assert reranker.max_length() == 512

    def test_env_override(self, monkeypatch):
        from services import reranker

        monkeypatch.setenv("RERANK_MAX_LENGTH", "1024")
        assert reranker.max_length() == 1024

    def test_illegal_value_falls_back(self, monkeypatch):
        from services import reranker

        monkeypatch.setenv("RERANK_MAX_LENGTH", "not-a-number")
        assert reranker.max_length() == 512


# ----------------------------------------------------------------------
# 补充: min_rerank_score 配置默认值
# ----------------------------------------------------------------------

class TestMinRerankScoreConfig:
    def test_settings_default_is_none(self, monkeypatch):
        """默认 **None = 不设阈值(不过滤)**。

        旧默认 0.0 是有害的: bge logit 随 query 口语化程度整体下移, 实测 24 条
        口语化 query 里 12 条的正确答案精排分为负 (多为排名第 1), 设 0.0 会把
        recall@3 从 0.9167 打到 0.5000。
        """
        from config.settings import AppSettings

        monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
        assert AppSettings().min_rerank_score is None

    def test_env_override(self, monkeypatch):
        from config.settings import AppSettings

        monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "1.5")
        assert AppSettings().min_rerank_score == 1.5

    def test_settings_explicit_zero_still_works(self, monkeypatch):
        """显式设 0.0 仍按数值生效 (机制保留, 便于实验)。"""
        from config.settings import AppSettings

        monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "0.0")
        assert AppSettings().min_rerank_score == 0.0

    def test_settings_illegal_env_falls_back_to_none(self, monkeypatch):
        """非法值告警并回退 None(不静默启用一个没校准的阈值)。"""
        from config.settings import AppSettings

        monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "abc")
        assert AppSettings().min_rerank_score is None

    def test_hybrid_retriever_default_is_none(self, monkeypatch):
        from services import hybrid_retriever as hr_mod

        monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
        assert hr_mod.min_rerank_score_default() is None

    def test_hybrid_retriever_reads_env_default(self, monkeypatch):
        from services import hybrid_retriever as hr_mod

        monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "2.5")
        assert hr_mod.min_rerank_score_default() == 2.5

    def test_hybrid_retriever_illegal_env_falls_back(self, monkeypatch):
        from services import hybrid_retriever as hr_mod

        monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "abc")
        assert hr_mod.min_rerank_score_default() is None
