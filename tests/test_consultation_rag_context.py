"""
RAG 检索带对话上下文 (consultation_rag_context) 单元测试

覆盖改动:
1. KnowledgeRetriever.search_knowledge 默认(无历史)行为不变
2. 传入 query_context 时,历史拼进 effective_query 一起送 FAISS
3. ConsultationProcessor.process_consultation_stream 把 query_context 透传给 RAG
4. 历史为空时 query_context 为空 → effective_query 与原始 query 相同

本文件只验证 query_context 拼接与透传,与 query 变换无关,故 autouse 固定
QUERY_TRANSFORM=off:底层收到的检索串恒为 [effective_query] 单条,断言不受
变换规则增删影响(变换本身由 tests/test_query_transform.py 覆盖)。

异步测试统一用 _run(coro) 同步包装(mock 知识库,不触发真实检索/LLM)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_query_transform_off(monkeypatch):
    """固定 off 档,让本文件的断言只反映 query_context 拼接逻辑。"""
    monkeypatch.setattr(settings, "query_transform_mode", "off")


def _run(coro):
    return asyncio.run(coro)


def _make_retriever(mock_search):
    """构造 KnowledgeRetriever,用 mock 顶替 knowledge_service.search_multi 以隔离真实检索。

    固定 retrieval_mode='dense': 本测试只验证 query_context 透传逻辑,
    不受 .env RETRIEVAL_MODE 影响(hybrid 时 mock 的是另一条路径)。

    底层检索入口是 search_multi(多查询检索, search() 是其单查询薄包装),
    因为这会随 query 变换档位收到 1~N 条检索串。
    """
    from agents.consultant.knowledge_retriever import KnowledgeRetriever
    retriever = KnowledgeRetriever()
    retriever.retrieval_mode = "dense"  # 只测 dense 透传
    mock_service = MagicMock()
    mock_service.search_multi = mock_search
    retriever.knowledge_service = mock_service
    retriever.kb_initialized = True  # 跳过真实初始化,直接走检索
    return retriever


class TestSearchKnowledgeQueryContext:
    def test_default_no_history_passes_query_only(self):
        """默认调用(不传 query_context / 不传过滤参数)应原样把 query 送进 FAISS,行为不变。

        P0-1 起 subject / grade_level 会显式透传(此处均为 None = 全量检索)。
        """
        retriever = _make_retriever(AsyncMock(return_value=[{"content": "doc"}]))
        docs = _run(retriever.search_knowledge("计算薄弱", top_k=3))
        assert docs  # 检索结果正常透传
        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["计算薄弱"], top_k=3, subject=None, grade_level=None
        )

    def test_with_query_context_splices_history_before_query(self):
        """传 query_context 时,历史应拼在 query 前面构成 effective_query。"""
        retriever = _make_retriever(AsyncMock(return_value=[{"content": "doc"}]))
        _run(retriever.search_knowledge(
            "计算薄弱", top_k=3, query_context="用户: 初二学生,数学不好"
        ))
        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["用户: 初二学生,数学不好\n计算薄弱"], top_k=3,
            subject=None, grade_level=None,
        )

    def test_empty_history_keeps_query_unchanged(self):
        """query_context 为空串时,effective_query 应与原始 query 相同。"""
        retriever = _make_retriever(AsyncMock(return_value=[]))
        _run(retriever.search_knowledge("计算薄弱", query_context=""))
        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["计算薄弱"], top_k=3, subject=None, grade_level=None
        )


class TestConsultationProcessorQueryContext:
    async def _collect(self, async_gen):
        out = []
        async for tok in async_gen:
            out.append(tok)
        return out

    def _make_processor(self, retriever, response_text="回答"):
        from agents.consultant.consultation_processor import ConsultationProcessor

        generator = MagicMock()
        async def _stream_gen(_u, _d):
            for t in response_text:
                yield t
        generator.generate_response_stream = _stream_gen

        classifier = MagicMock()

        proc = ConsultationProcessor(
            knowledge_retriever=retriever,
            consultation_classifier=classifier,
            response_generator=generator,
        )
        # 屏蔽用户行为记录,保持测试聚焦且不触碰任何存储
        proc._record_consultation_behavior = AsyncMock()
        return proc

    def test_stream_passes_query_context_to_rag(self):
        """query_context 传入后,RAG 检索应收到带历史拼接的查询 + 抽出的学科/年级。"""
        retriever = _make_retriever(AsyncMock(return_value=[{"content": "数学提分方案"}]))
        proc = self._make_processor(retriever)

        gen = proc.process_consultation_stream(
            "计算薄弱", "sess-1", query_context="用户: 初二,数学不行"
        )
        out = _run(self._collect(gen))
        assert "".join(out) == "回答"

        # 未注入 profile_provider → 不兜底;学科/年级从 query_context 抽出
        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["用户: 初二,数学不行\n计算薄弱"], top_k=3,
            subject="数学", grade_level="初中",
        )

    def test_stream_empty_query_context_keeps_query(self):
        """未传 query_context 时,RAG 检索仍收到原 query(向后兼容)。"""
        retriever = _make_retriever(AsyncMock(return_value=[{"content": "doc"}]))
        proc = self._make_processor(retriever)

        _run(self._collect(proc.process_consultation_stream("计算薄弱", "sess-1")))

        retriever.knowledge_service.search_multi.assert_awaited_once_with(
            ["计算薄弱"], top_k=3, subject=None, grade_level=None
        )


class TestConsultantAgentRagQueryContext:
    def test_build_rag_query_context_empty_when_no_history(self, monkeypatch):
        """无历史/仅占位文案时,_build_rag_query_context 应返回空串。"""
        monkeypatch.setattr("config.model_provider.create_chat_model", lambda **kw: MagicMock())
        from agents.consultant import knowledge_retriever as kr_mod
        from agents.consultant import consultation_classifier as cc_mod
        from agents.consultant import response_generator as rg_mod
        monkeypatch.setattr(kr_mod, "KnowledgeRetriever", MagicMock())
        monkeypatch.setattr(cc_mod, "ConsultationClassifier", MagicMock())
        monkeypatch.setattr(rg_mod, "ResponseGenerator", MagicMock())

        from agents.consultant_agent import ConsultantAgent
        agent = ConsultantAgent(session_id="rag-sess")
        # ContextManager 已在上文构造(consult_stream 前不 build,chat 历史为空)
        assert agent._build_rag_query_context() == ""

    def test_build_rag_query_context_returns_history(self, monkeypatch):
        """有历史写入 ConversationContext 时,应返回多行「用户/顾问: 内容」文本。"""
        monkeypatch.setattr("config.model_provider.create_chat_model", lambda **kw: MagicMock())
        from agents.consultant import knowledge_retriever as kr_mod
        from agents.consultant import consultation_classifier as cc_mod
        from agents.consultant import response_generator as rg_mod
        monkeypatch.setattr(kr_mod, "KnowledgeRetriever", MagicMock())
        monkeypatch.setattr(cc_mod, "ConsultationClassifier", MagicMock())
        monkeypatch.setattr(rg_mod, "ResponseGenerator", MagicMock())

        from agents.consultant_agent import ConsultantAgent
        agent = ConsultantAgent(session_id="rag-sess-2")

        from agents.context.contexts.conversation import ConversationContext
        conv = next(
            c for c in agent.context_manager.contexts
            if isinstance(c, ConversationContext)
        )
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage
        history = conv.get_history(agent.session_id)
        history.add_message(HumanMessage(content="我家孩子初二,数学不好"))
        history.add_message(AIMessage(content="收到,会针对初二数学答疑"))

        text = agent._build_rag_query_context()
        assert "初二" in text
        assert "收到,会针对初二数学答疑" in text