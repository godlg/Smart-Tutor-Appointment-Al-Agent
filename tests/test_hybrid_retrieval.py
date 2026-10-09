"""混合检索 (BM25 + Dense + RRF + ms-marco 精排) 单元测试。

覆盖:
1. Bm25Index 中文分词 + 专有名词命中 + 无命中/空库
2. RRF 融合公式与跨路融合顺序
3. HybridRetriever 无 filter / metadata filter / category 过滤
4. 精排路径(真实模型) + 模型缺失/异常回退 RRF
5. reranker 模块打分正确性(真实模型, 模型文件缺失时跳过)

Embedding 用确定性 fake(sha256→16 维), 不依赖远程 API。
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ANY_RERANKER_MODEL = any(
    (PROJECT_ROOT / p).exists()
    for p in (
        "cache/bge-reranker-base/onnx/model_quantized.onnx",
        "cache/ms-marco-onnx/onnx/model.onnx",
    )
)


def _run(coro):
    return asyncio.run(coro)


def _fake_embed(text: str) -> List[float]:
    """sha256→16 维确定性向量, 相似文本向量相近(不归一化, 内积仍可排序)。"""
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
    """种子 4 条: 学科/年级/类别组合, 专有名词突出便于 BM25 验证。"""
    ids = []
    ids.append(repo.add_document(
        content="高中数学 12 周提分方案: 函数、几何、数列、概率四大模块, 含韦达定理专题。",
        category="course_intro", keywords=["高中", "数学", "高考", "韦达定理"],
        subject="数学", grade_level="高中", doc_type="course_intro",
    ))
    ids.append(repo.add_document(
        content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
        category="course_intro", keywords=["小学", "阅读", "绘本"],
        subject="语文", grade_level="小学", doc_type="course_intro",
    ))
    ids.append(repo.add_document(
        content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
        category="subject_method", keywords=["初中", "数学", "一元二次方程", "韦达定理"],
        subject="数学", grade_level="初中", doc_type="subject_method",
    ))
    ids.append(repo.add_document(
        content="如何帮孩子选家教老师: 看学科匹配、年级段、教学风格, 建议先试讲。",
        category="parent_guide", keywords=["家长", "老师", "试讲"],
        subject=None, grade_level=None, doc_type="parent_guide",
    ))
    return ids


@pytest.fixture
def hybrid_env():
    """内存库 + 种子数据 + 已初始化的 HybridRetriever。"""
    from services.knowledge_service import KnowledgeService
    from services.hybrid_retriever import HybridRetriever

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    _seed_corpus(svc.db)
    _run(svc.initialize())  # 构建 FAISS 索引 + 落库 embedding

    retriever = HybridRetriever(svc, rrf_k=60, candidates=20, rerank=False)
    return {"svc": svc, "retriever": retriever}


# ----------------------------------------------------------------------
# tokenize
# ----------------------------------------------------------------------

class TestTokenize:
    def test_chinese_terms(self):
        from services.bm25_index import tokenize

        tokens = tokenize("初二数学 分数运算 薄弱")
        assert "数学" in tokens and "分数" in tokens and "运算" in tokens
        # 停用词被滤掉
        assert "的" not in tokenize("分数的运算方法")

    def test_english_and_digits_kept(self):
        from services.bm25_index import tokenize

        tokens = tokenize("Python 3.10 async generator")
        assert "python" in tokens and "async" in tokens and "generator" in tokens


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------

class TestBm25Index:
    def test_build_and_search_hits_special_term(self):
        from services.bm25_index import Bm25Index

        docs = [
            {"id": 1, "content": "初中数学 一元二次方程 韦达定理", "keywords": []},
            {"id": 2, "content": "小学阅读 经典绘本 词汇造句", "keywords": []},
        ]
        idx = Bm25Index().build(docs)
        results = idx.search("韦达定理", top_n=5)
        assert results, "BM25 应命中含 '韦达定理' 的文档"
        assert results[0][0] == 1  # doc_id=1 排第一

    def test_search_no_hit_returns_empty(self):
        from services.bm25_index import Bm25Index

        idx = Bm25Index().build([{"id": 1, "content": "高中英语语法"}])
        assert idx.search("量子物理", top_n=5) == []

    def test_empty_build_returns_empty(self):
        from services.bm25_index import Bm25Index

        assert Bm25Index().build([]).search("任何", top_n=5) == []
        assert not Bm25Index().built


# ----------------------------------------------------------------------
# RRF
# ----------------------------------------------------------------------

class TestRrfMerge:
    def test_same_ranking_preserved(self):
        from services.hybrid_retriever import rrf_merge

        merged = rrf_merge([[(1, 1), (2, 2)], [(1, 1), (2, 2)]], k=60)
        assert [did for did, _ in merged] == [1, 2]

    def test_cross_ranking_fusion(self):
        """doc 1 双路都第一 → 2/(61); doc 2 一路第一一路第十 → 1/61+1/70, 应仍居前。"""
        from services.hybrid_retriever import rrf_merge

        merged = rrf_merge([[(1, 1), (2, 2)], [(1, 1), (2, 10)]], k=60)
        dids = [did for did, _ in merged]
        assert dids.index(1) < dids.index(2)

    def test_float_score(self):
        from services.hybrid_retriever import rrf_merge

        merged = rrf_merge([[(1, 1)]], k=60)
        assert abs(merged[0][1] - 1.0 / 61.0) < 1e-9


# ----------------------------------------------------------------------
# HybridRetriever
# ----------------------------------------------------------------------

class TestHybridSearch:
    def test_structure_and_top_k(self, hybrid_env):
        results = _run(hybrid_env["retriever"].search("韦达定理", top_k=3))
        assert 1 <= len(results) <= 3
        for doc in results:
            assert doc["content"]
            assert "score" in doc and "rank" in doc and "category" in doc

    def test_returns_top_ranked_first(self, hybrid_env):
        results = _run(hybrid_env["retriever"].search("韦达定理 初中数学", top_k=3))
        assert results and results[0]["rank"] == 1
        # 韦达定理文档(含关键词)应出现在结果里
        contents = [d["content"] for d in results]
        assert any("韦达定理" in c for c in contents)

    def test_respects_metadata_filter(self, hybrid_env):
        """subject 硬过滤必须生效:非本学科文档一律挡掉。

        P0-2 起 subject 为 NULL 的跨学科文档(parent_guide)有豁免,允许进候选;
        除它以外,结果里的学科只能是「数学」。
        """
        results = _run(hybrid_env["retriever"].search(
            "课程方案", top_k=5, subject="数学"
        ))
        assert results
        for doc in results:
            assert doc["subject"] in ("数学", None), (
                f"非数学学科不应出现在 subject=数学 的候选里: {doc['subject']!r}"
            )
        # 语文/小学 的文档必须被硬过滤挡掉
        assert not any(doc["subject"] == "语文" for doc in results)

    def test_category_filter_applied_after(self, hybrid_env):
        results = _run(hybrid_env["retriever"].search(
            "绘本 阅读", top_k=5, category="course_intro"
        ))
        assert results
        for doc in results:
            assert doc["category"] == "course_intro"

    def test_rerank_disabled_uses_rrf_score(self, hybrid_env):
        results = _run(hybrid_env["retriever"].search("高中数学 函数", top_k=2))
        assert results
        for doc in results:
            assert "rrf_score" in doc
            assert "rerank_score" not in doc  # rerank=False 不精排


# ----------------------------------------------------------------------
# 精排路径与回退
# ----------------------------------------------------------------------

class TestRerankPath:
    @pytest.mark.skipif(not _ANY_RERANKER_MODEL, reason="精排模型未下载(cache/)")
    def test_rerank_true_changes_score_field(self, hybrid_env):
        retriever = hybrid_env["retriever"]
        retriever.rerank = True
        results = _run(retriever.search("小学数学 绘本 阅读", top_k=2))
        assert results
        assert all("rerank_score" in d for d in results), "精排应写入 rerank_score"

    def test_rerank_unavailable_falls_back_to_rrf(self, hybrid_env, monkeypatch):
        """模型不可用(available=False) → 静默回退 RRF, 不抛异常。"""
        from services import hybrid_retriever as hr_mod

        monkeypatch.setattr(hr_mod.reranker, "available", lambda: False)
        retriever = hybrid_env["retriever"]
        retriever.rerank = True
        results = _run(retriever.search("韦达定理", top_k=3))
        assert results
        assert all("rerank_score" not in d for d in results)

    def test_rerank_exception_falls_back_to_rrf(self, hybrid_env, monkeypatch):
        """精排打分抛异常 → 回退 RRF, 检索仍成功。"""
        from services import hybrid_retriever as hr_mod

        def _boom(*a, **k):
            raise RuntimeError("模拟精排崩溃")

        monkeypatch.setattr(hr_mod.reranker, "available", lambda: True)
        monkeypatch.setattr(hr_mod.reranker, "rerank", _boom)
        retriever = hybrid_env["retriever"]
        retriever.rerank = True
        results = _run(retriever.search("高中数学 提分", top_k=3))
        assert results, "精排异常后应回退 RRF 并正常返回"


# ----------------------------------------------------------------------
# reranker 模块(真实模型)
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _ANY_RERANKER_MODEL, reason="精排模型未下载(cache/)")
class TestRerankerModule:
    def test_score_pair_relevant_gt_irrelevant(self):
        from services.reranker import score_pair

        relevant = score_pair(
            "初中数学 一元二次方程 韦达定理",
            "初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)。",
        )
        irrelevant = score_pair(
            "初中数学 一元二次方程 韦达定理",
            "小学英语 自然拼读 常见词练习。",
        )
        assert relevant > irrelevant, "相关对精排分应高于不相关对"

    def test_rerank_sorts_and_not_mutate(self):
        from services.reranker import rerank

        docs = [
            {"id": 1, "content": "小学英语 自然拼读 常见词练习。"},
            {"id": 2, "content": "初中数学 一元二次方程 韦达定理 判别式。"},
        ]
        out = rerank("韦达定理 一元二次方程", docs, top_k=2)
        assert out and out[0]["id"] == 2  # 相关文档排第一
        assert "rerank_score" in out[0]
        assert "rerank_score" not in docs[0]  # 原 dict 未污染
