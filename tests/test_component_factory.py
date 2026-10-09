"""可插拔组件层 (六类 Base + YAML + ComponentFactory) 单元测试。

覆盖 (对应任务 DoD):
1. YAML 加载 / 缺失回退默认 / 未知 type 报清晰错误
2. 工厂创建每类组件 (LLM/Embedding 只验证对象创建, 不真实调 API)
3. VectorStore 三选一 (dense / bm25 / hybrid) 各自可 search
4. Evaluator 对假 retriever 算 recall/mrr 正确
5. 演示用例: 工厂创建 hybrid → 真实检索命中
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

from services.components.base import QueryCase  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _fake_embed(text: str) -> List[float]:
    """sha256→16 维确定性向量 (与 test_hybrid_retrieval 相同风格)。"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input: 覆盖 text_embedding / knowledge_service / hybrid_retriever。"""
    from services import text_embedding, knowledge_service, hybrid_retriever

    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)
    monkeypatch.setattr(knowledge_service, "embed_input", _fake_embed)
    monkeypatch.setattr(hybrid_retriever, "embed_input", _fake_embed)


@pytest.fixture
def seeded_svc():
    """内存库 + 3 条种子文档 + 已初始化 (FAISS 索引就绪)。"""
    from services.knowledge_service import KnowledgeService

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    svc.db.add_document(
        content="高中数学 12 周提分方案: 函数、几何、数列、概率四大模块, 含韦达定理专题。",
        category="course_intro", keywords=["高中", "数学", "高考", "韦达定理"],
        subject="数学", grade_level="高中", doc_type="course_intro",
    )
    svc.db.add_document(
        content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
        category="subject_method", keywords=["初中", "数学", "一元二次方程", "韦达定理"],
        subject="数学", grade_level="初中", doc_type="subject_method",
    )
    svc.db.add_document(
        content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
        category="course_intro", keywords=["小学", "阅读", "绘本"],
        subject="语文", grade_level="小学", doc_type="course_intro",
    )
    _run(svc.initialize())
    return svc


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "components.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# 1. YAML 加载 / 默认值 / 未知 type 报错
# ----------------------------------------------------------------------

class TestYamlLoading:
    def test_load_default_yaml(self):
        """真实 config/components.yaml 应能加载出六类声明。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory()
        cfg = factory.config
        assert set(cfg) == {"llm", "embedding", "splitter", "vector_store", "reranker", "evaluator"}
        assert cfg["vector_store"]["type"] == "hybrid"
        assert cfg["splitter"]["chunk_size"] == 500

    def test_missing_yaml_file_falls_back_to_defaults(self, tmp_path):
        """YAML 文件不存在 → 全部走默认实现, 不抛错。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory(yaml_path=tmp_path / "does_not_exist.yaml")
        cfg = factory.config
        assert cfg["llm"]["type"] == "openai_compatible"
        assert cfg["splitter"]["type"] == "recursive"
        assert cfg["vector_store"]["type"] == "hybrid"
        assert cfg["reranker"]["type"] == "bge"
        assert cfg["evaluator"]["type"] == "recall_at_k"

    def test_partial_yaml_missing_component(self, tmp_path):
        """YAML 只声明部分类 → 缺失类回退默认实现。"""
        from services.components.factory import ComponentFactory

        p = _write_yaml(tmp_path, "components:\n  vector_store: { type: bm25 }\n")
        factory = ComponentFactory(yaml_path=p)
        cfg = factory.config
        assert cfg["vector_store"]["type"] == "bm25"
        assert cfg["llm"]["type"] == "openai_compatible"  # 未声明 → 默认
        assert cfg["splitter"]["type"] == "recursive"

    def test_unknown_type_raises_clear_error(self, tmp_path):
        """未知 type 应报 ValueError 且错误信息含可用 type 列表。"""
        from services.components.factory import ComponentFactory

        p = _write_yaml(tmp_path, "components:\n  vector_store: { type: nonexistent }\n")
        factory = ComponentFactory(yaml_path=p)
        with pytest.raises(ValueError, match="nonexistent"):
            factory.get_vector_store()


# ----------------------------------------------------------------------
# 2. 工厂创建每类组件 (LLM/Embedding 只验证对象创建)
# ----------------------------------------------------------------------

class TestFactoryCreatesComponents:
    def test_create_llm(self, tmp_path):
        """LLM 对象可创建 (不真实调 API); 参数 temperature 透传。"""
        from services.components.factory import ComponentFactory

        p = _write_yaml(tmp_path, "components:\n  llm: { type: openai_compatible, temperature: 0.7 }\n")
        llm = ComponentFactory(yaml_path=p).get_llm()
        from services.components.lightweight import OpenAICompatibleLLM

        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.temperature == 0.7

    def test_create_embedding(self):
        """Embedding 对象可创建, 具备 embed 批量接口。"""
        from services.components.factory import ComponentFactory

        emb = ComponentFactory().get_embedding()
        from services.components.lightweight import OpenAICompatibleEmbedding

        assert isinstance(emb, OpenAICompatibleEmbedding)
        # 只验证批量签名存在 (真实 embed 需 API, 不在此调)
        assert callable(emb.embed)

    def test_create_splitter(self):
        """Splitter 可创建并完成分块 (纯本地逻辑)。"""
        from services.components.factory import ComponentFactory

        splitter = ComponentFactory().get_splitter()
        text = "高中数学 12 周提分方案: 函数、几何、数列、概率四大模块。" * 30
        chunks = splitter.split(text)
        assert len(chunks) > 1, "长文本应被切成多块"
        assert all(isinstance(c, str) and c for c in chunks)

    def test_create_reranker_bge_and_none(self):
        """reranker 支持 bge / none 两型 (bge 缺失模型时回退, 不抛)。"""
        from services.components.factory import ComponentFactory

        bge = ComponentFactory().get_reranker()
        docs = [{"id": 1, "content": "韦达定理"}, {"id": 2, "content": "绘本阅读"}]
        out = bge.rerank("韦达定理", docs, top_k=2)
        assert len(out) == 2 and out[0]["id"] == 1  # 无模型时原样返回前 top_k

        from services.components.factory import ComponentFactory as CF

        f2 = CF()
        f2._config = {"reranker": {"type": "none"}}
        noop = f2.get_reranker()
        out2 = noop.rerank("x", [{"id": 9}], top_k=1)
        assert out2 == [{"id": 9}]


# ----------------------------------------------------------------------
# 3. VectorStore 三选一各自可 search
# ----------------------------------------------------------------------

class TestVectorStoreThreeKinds:
    def test_dense_search(self, seeded_svc):
        """type=dense → 走 KnowledgeService.search, 检索命中相关文档。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory(knowledge_service=seeded_svc)
        factory._config = {"vector_store": {"type": "dense"}}
        vs = factory.get_vector_store()
        results = vs.search("韦达定理 初中数学", top_k=3)
        assert results
        assert results[0]["rank"] == 1
        assert "score" in results[0]
        contents = [r["content"] for r in results]
        assert any("韦达定理" in c for c in contents)

    def test_bm25_search(self, seeded_svc):
        """type=bm25 → 走 Bm25Index, 专有名词精确命中。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory(knowledge_service=seeded_svc)
        factory._config = {"vector_store": {"type": "bm25"}}
        vs = factory.get_vector_store()
        vs.add(seeded_svc.get_all_documents())
        results = vs.search("一元二次方程", top_k=3)
        assert results
        assert results[0]["id"] == 2  # 含"一元二次方程"的 doc_id=2 排第一
        assert results[0]["rank"] == 1

    def test_hybrid_search(self, seeded_svc):
        """type=hybrid → 走 HybridRetriever, RRF 融合检索命中。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory(knowledge_service=seeded_svc)
        factory._config = {"vector_store": {"type": "hybrid", "rrf_k": 60}}
        vs = factory.get_vector_store()
        results = vs.search("韦达定理 初中数学", top_k=3)
        assert results
        assert results[0]["rank"] == 1
        assert "score" in results[0]
        contents = [r["content"] for r in results]
        assert any("韦达定理" in c for c in contents)

    def test_vector_store_requires_knowledge_service(self):
        """未传 knowledge_service 时创建 vector_store 应报清晰错误。"""
        from services.components.factory import ComponentFactory

        factory = ComponentFactory()
        with pytest.raises(ValueError, match="knowledge_service"):
            factory.get_vector_store()


# ----------------------------------------------------------------------
# 4. Evaluator 对假 retriever 算 recall/mrr 正确
# ----------------------------------------------------------------------

class _FakeRetriever:
    """可控假 retriever: search(query, top_k) 返回固定排序。"""

    def __init__(self, ranked_by_query: dict):
        self._ranked = ranked_by_query

    def search(self, query: str, top_k: int = 3):
        return [
            {"id": doc_id, "rank": rank, "score": 1.0}
            for rank, doc_id in enumerate(self._ranked[query][:top_k], start=1)
        ]


class TestEvaluatorRecallAtK:
    def test_perfect_recall_and_mrr(self):
        """每 query 首个即命中 → recall=1.0, mrr=1.0。"""
        from services.components.factory import ComponentFactory

        cases = [
            QueryCase(query="q1", expected_doc_ids=[1]),
            QueryCase(query="q2", expected_doc_ids=[5]),
        ]
        retriever = _FakeRetriever({"q1": [1, 2], "q2": [5, 6]})
        evaluator = ComponentFactory(evaluator_cases=cases).get_evaluator()
        metrics = evaluator.evaluate(retriever)

        assert metrics["recall_at_k"] == 1.0
        assert metrics["mrr"] == 1.0
        assert len(metrics["case_results"]) == 2

    def test_partial_recall_and_mrr(self):
        """q1 期望 {1,2}, 只命中 1 (第 2 位) → case recall=0.5, mrr=0.5;
        q2 未命中 → 0; 总指标为两 case 均值 0.25。"""
        from services.components.factory import ComponentFactory

        cases = [
            QueryCase(query="q1", expected_doc_ids=[1, 2]),
            QueryCase(query="q2", expected_doc_ids=[9]),
        ]
        retriever = _FakeRetriever({"q1": [3, 1], "q2": [7]})
        evaluator = ComponentFactory(evaluator_cases=cases).get_evaluator()
        metrics = evaluator.evaluate(retriever)

        assert metrics["recall_at_k"] == pytest.approx(0.25)
        assert metrics["mrr"] == pytest.approx(0.25)  # (0.5 + 0) / 2
        assert metrics["case_results"][0]["recall"] == pytest.approx(0.5)

    def test_second_rank_hit_mrr(self):
        """期望 doc 在第 2 位 → 该 case mrr = 1/2。"""
        from services.components.factory import ComponentFactory

        cases = [QueryCase(query="q1", expected_doc_ids=[2])]
        retriever = _FakeRetriever({"q1": [5, 2, 1]})
        evaluator = ComponentFactory(evaluator_cases=cases).get_evaluator()
        metrics = evaluator.evaluate(retriever)

        assert metrics["mrr"] == pytest.approx(0.5)
        case = metrics["case_results"][0]
        assert case["hit_ids"] == [2]

    def test_empty_expected_skipped(self):
        """没有期望 doc 的 case 不参与指标计算。"""
        from services.components.factory import ComponentFactory

        cases = [QueryCase(query="q1", expected_doc_ids=[])]
        retriever = _FakeRetriever({"q1": [1]})
        evaluator = ComponentFactory(evaluator_cases=cases).get_evaluator()
        metrics = evaluator.evaluate(retriever)

        assert metrics["recall_at_k"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["case_results"] == []


# ----------------------------------------------------------------------
# 5. 演示用例: 工厂创建 hybrid → 真实检索命中 + 评测
# ----------------------------------------------------------------------

class TestFactoryDemoFlow:
    def test_factory_hybrid_search_and_evaluate(self, seeded_svc):
        """端到端: YAML 默认 hybrid → 检索命中 → Evaluator 打分。"""
        from services.components.factory import ComponentFactory

        cases = [
            QueryCase(query="韦达定理 初中数学", expected_doc_ids=[2]),
            QueryCase(query="小学 绘本 阅读", expected_doc_ids=[3]),
        ]
        factory = ComponentFactory(knowledge_service=seeded_svc, evaluator_cases=cases)

        vs = factory.get_vector_store()
        results = vs.search("韦达定理 初中数学", top_k=3)
        assert results, "hybrid 检索应有命中"
        assert any("韦达定理" in r["content"] for r in results)

        evaluator = factory.get_evaluator()
        metrics = evaluator.evaluate(vs)
        assert metrics["recall_at_k"] > 0.0, "至少一个 query 命中"
        assert metrics["mrr"] > 0.0
        assert len(metrics["case_results"]) == 2
