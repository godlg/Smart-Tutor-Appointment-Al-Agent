"""
Consultant Agent 单元测试 (补1)

覆盖 agents/consultant/* 纯逻辑组件 (PromptBuilder, ConsultationProcessor,
ConsultationClassifier),通过 mock LLM / mock retriever 隔离外部依赖。

被测组件不发起任何外部 API 调用,跑得快、可在 CI 上反复跑。
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

class _FakeChatModel:
    """最小化的 BaseChatModel 替身:只支持 ainvoke 与 astream。

    LangChain BaseChatModel 是个庞大的协议类;我们的代码只用 ainvoke + content 属性,
    所以只需 mock 这两点。
    """

    def __init__(self, content: str = "YES", tokens: List[str] = None):
        self._content = content
        self._tokens = tokens or ["这是", "一个", "回答"]

    async def ainvoke(self, messages, **kwargs):
        msg = MagicMock()
        msg.content = self._content
        return msg

    async def astream(self, messages, **kwargs):
        from langchain_core.messages import AIMessageChunk
        for t in self._tokens:
            yield AIMessageChunk(content=t)


@pytest.fixture
def fake_llm_yes():
    return _FakeChatModel(content="YES")


@pytest.fixture
def fake_llm_no():
    return _FakeChatModel(content="NO")


@pytest.fixture
def fake_llm_stream():
    return _FakeChatModel(tokens=["高中", "数学", "提分"])


@pytest.fixture
def fake_knowledge_docs() -> List[Dict[str, Any]]:
    return [
        {
            "id": 1,
            "content": "高中数学 12 周提分方案",
            "category": "course_intro",
            "score": 0.95,
        },
        {
            "id": 2,
            "content": "数学思维训练方法",
            "category": "subject_method",
            "score": 0.80,
        },
    ]


# ----------------------------------------------------------------------
# PromptBuilder: 纯字符串模板
# ----------------------------------------------------------------------

class TestPromptBuilder:
    """PromptBuilder 不依赖任何外部资源,纯字符串拼装,单测性价比最高。"""

    def test_build_classification_prompt_includes_user_input(self):
        from agents.consultant.prompt_builder import PromptBuilder
        prompt = PromptBuilder().build_classification_prompt("如何帮孩子预习数学?")
        assert "如何帮孩子预习数学?" in prompt
        assert "YES" in prompt  # 模板要求回答 YES/NO

    def test_build_classification_prompt_mentions_tutoring_scope(self):
        """模板应明确说明何为咨询类问题,避免误判。"""
        from agents.consultant.prompt_builder import PromptBuilder
        prompt = PromptBuilder().build_classification_prompt("test")
        assert "K12" in prompt or "家教" in prompt
        assert "YES" in prompt and "NO" in prompt

    def test_build_consultation_prompt_with_no_docs_says_no_match(self):
        """空知识库时,prompt 应明确告诉 LLM '没找到,请基于常识'。"""
        from agents.consultant.prompt_builder import PromptBuilder
        prompt = PromptBuilder().build_consultation_prompt("test question", [])
        assert "test question" in prompt
        assert "没有找到" in prompt or "常识" in prompt

    def test_build_consultation_prompt_with_docs_numbers_them(self):
        """多条知识应带来源编号(资料1/2/3),LLM 引用时不会乱。"""
        from agents.consultant.prompt_builder import PromptBuilder
        docs = [
            {"content": "第一段", "category": "faq"},
            {"content": "第二段", "category": "faq"},
            {"content": "第三段", "category": "faq"},
        ]
        prompt = PromptBuilder().build_consultation_prompt("Q", docs)
        assert "[资料1|faq] 第一段" in prompt
        assert "[资料2|faq] 第二段" in prompt
        assert "[资料3|faq] 第三段" in prompt

    def test_build_consultation_prompt_contains_system_role(self):
        """system_prompt 应进入 prompt,体现 agent 的角色约束。"""
        from agents.consultant.prompt_builder import PromptBuilder
        prompt = PromptBuilder().build_consultation_prompt("Q", [])
        assert "家教" in prompt or "K12" in prompt
        # 用户问题要在最末
        assert prompt.rstrip().endswith("请回答用户的问题。")


# ----------------------------------------------------------------------
# ConsultationClassifier: mock LLM 验证 YES/NO 解析
# ----------------------------------------------------------------------

class TestConsultationClassifier:
    """只测 LLM 响应解析逻辑 (剥除实际推理)。"""

    async def _classify(self, fake_llm, user_input: str) -> bool:
        from agents.consultant.consultation_classifier import ConsultationClassifier
        clf = ConsultationClassifier(llm=fake_llm)
        return await clf.is_consultation_related(user_input)

    def test_yes_response_is_consultation_related(self, fake_llm_yes):
        assert _run(self._classify(fake_llm_yes, "如何辅导孩子数学?")) is True

    def test_no_response_is_not_related(self, fake_llm_no):
        assert _run(self._classify(fake_llm_no, "今天天气怎么样?")) is False

    def test_lowercase_yes_is_still_accepted(self):
        """LLM 可能回 'yes' (小写),应该被正确识别。"""
        fake = _FakeChatModel(content="yes")
        assert _run(self._classify(fake, "x")) is True

    def test_lowercase_no_is_still_accepted(self):
        fake = _FakeChatModel(content="no")
        assert _run(self._classify(fake, "x")) is False

    def test_whitespace_around_answer_is_stripped(self):
        """LLM 可能回 ' YES \n',应被 trim 后识别。"""
        fake = _FakeChatModel(content="  YES  \n")
        assert _run(self._classify(fake, "x")) is True

    def test_garbage_response_returns_false(self):
        """LLM 返回既不是 YES 也不是 NO 的内容时,严格匹配返回 False。"""
        fake = _FakeChatModel(content="我也不知道")  # 既不是 YES 也不是 NO
        assert _run(self._classify(fake, "x")) is False

    def test_exception_during_invoke_defaults_to_related(self):
        """LLM 抛异常时,默认 True (避免因为网络/限流误杀咨询请求)。

        这是与 garbage response 不同的策略:
        - garbage 走 LLM 推理结果 (False) — 用户得到了 LLM 的回答
        - exception 走降级 (True) — 宁愿误判为咨询,不让用户请求被静默丢弃
        """
        broken = MagicMock()
        broken.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        from agents.consultant.consultation_classifier import ConsultationClassifier
        clf = ConsultationClassifier(llm=broken)
        assert _run(clf.is_consultation_related("x")) is True


# ----------------------------------------------------------------------
# ConsultationProcessor: mock retriever + generator 验证编排
# ----------------------------------------------------------------------

class TestConsultationProcessor:
    """验证 process_consultation / handle_unrelated_request 的协作流程。"""

    def _make_processor(self, retriever_docs, response_text):
        from agents.consultant.consultation_processor import ConsultationProcessor
        retriever = MagicMock()
        retriever.search_knowledge = AsyncMock(return_value=retriever_docs)

        generator = MagicMock()
        generator.generate_response = AsyncMock(return_value=response_text)
        generator.create_unrelated_message = MagicMock(return_value="[unrelated msg]")

        async def _stream_gen(_u, _d):
            for t in response_text.split("|"):
                yield t
        generator.generate_response_stream = _stream_gen

        classifier = MagicMock()
        return ConsultationProcessor(
            knowledge_retriever=retriever,
            consultation_classifier=classifier,
            response_generator=generator,
        )

    def test_process_consultation_searches_then_generates(self):
        """正常路径: 检索 → 生成 → 返回文本。"""
        docs = [{"content": "doc1"}, {"content": "doc2"}]
        proc = self._make_processor(retriever_docs=docs, response_text="最终回答")
        out = _run(proc.process_consultation("Q"))
        assert out == "最终回答"
        proc.knowledge_retriever.search_knowledge.assert_awaited_once_with("Q", top_k=3)
        proc.response_generator.generate_response.assert_awaited_once_with("Q", docs)

    def test_process_consultation_with_no_docs_still_responds(self):
        """检索为空时,生成器应拿到 [] 并返回兜底回答(不抛错)。"""
        proc = self._make_processor(retriever_docs=[], response_text="兜底")
        out = _run(proc.process_consultation("未知问题"))
        assert out == "兜底"

    def test_handle_unrelated_resets_shared_state_to_classify(self):
        """无关请求进入时应把 shared_state 重置为 CLASSIFY,下一轮重新分类。"""
        from config.constants import FlowState
        proc = self._make_processor(retriever_docs=[], response_text="")
        state = MagicMock()
        state.value = FlowState.CONSULTATION  # 假设当前在咨询中
        gen = proc.handle_unrelated_request("今天天气", unrelated_callback=None, shared_state=state)
        chunks = _run(self._collect(gen))
        assert state.value == FlowState.CLASSIFY
        assert "[unrelated msg]" in chunks

    @staticmethod
    async def _collect(async_gen):
        out = []
        async for tok in async_gen:
            out.append(tok)
        return out

    def test_handle_unrelated_invokes_callback_when_provided(self):
        """无关请求应被转给 callback 处理(预约/调课等)。"""
        proc = self._make_processor(retriever_docs=[], response_text="")

        async def _cb(user_input):
            for tok in ["调课:", user_input]:
                yield tok

        gen = proc.handle_unrelated_request(
            "请帮我调课", unrelated_callback=_cb, shared_state=None,
        )
        chunks = _run(self._collect(gen))
        assert chunks == ["[unrelated msg]", "调课:", "请帮我调课"]


# ----------------------------------------------------------------------
# ConsultantAgent: session_id 与状态管理
# ----------------------------------------------------------------------

class TestConsultantAgent:
    """只测不依赖 LLM 的部分: session_id 初始化、回调注册、状态 setter。"""

    def test_default_session_id_is_uuid(self, monkeypatch):
        """不传 session_id 时应自动生成 UUID(用于会话隔离)。"""
        # 屏蔽 __init__ 末尾的 LLM 初始化 + 子组件构造
        monkeypatch.setattr(
            "config.model_provider.create_chat_model",
            lambda **kw: MagicMock(),
        )
        # 屏蔽子组件,避免构造时依赖更深层的服务
        from agents.consultant import knowledge_retriever as kr_mod
        from agents.consultant import consultation_classifier as cc_mod
        from agents.consultant import response_generator as rg_mod

        monkeypatch.setattr(kr_mod, "KnowledgeRetriever", MagicMock())
        monkeypatch.setattr(cc_mod, "ConsultationClassifier", MagicMock())
        monkeypatch.setattr(rg_mod, "ResponseGenerator", MagicMock())

        from agents.consultant_agent import ConsultantAgent
        agent = ConsultantAgent()
        assert agent.session_id is not None
        assert len(agent.session_id) > 0
        # UUID 长度恒为 36
        assert len(agent.session_id) == 36

    def test_explicit_session_id_is_preserved(self, monkeypatch):
        monkeypatch.setattr("config.model_provider.create_chat_model", lambda **kw: MagicMock())
        from agents.consultant import knowledge_retriever as kr_mod
        from agents.consultant import consultation_classifier as cc_mod
        from agents.consultant import response_generator as rg_mod
        monkeypatch.setattr(kr_mod, "KnowledgeRetriever", MagicMock())
        monkeypatch.setattr(cc_mod, "ConsultationClassifier", MagicMock())
        monkeypatch.setattr(rg_mod, "ResponseGenerator", MagicMock())

        from agents.consultant_agent import ConsultantAgent
        agent = ConsultantAgent(session_id="custom-sess-001")
        assert agent.session_id == "custom-sess-001"

    def test_set_shared_state_and_callback(self, monkeypatch):
        monkeypatch.setattr("config.model_provider.create_chat_model", lambda **kw: MagicMock())
        from agents.consultant import knowledge_retriever as kr_mod
        from agents.consultant import consultation_classifier as cc_mod
        from agents.consultant import response_generator as rg_mod
        monkeypatch.setattr(kr_mod, "KnowledgeRetriever", MagicMock())
        monkeypatch.setattr(cc_mod, "ConsultationClassifier", MagicMock())
        monkeypatch.setattr(rg_mod, "ResponseGenerator", MagicMock())

        from agents.consultant_agent import ConsultantAgent
        agent = ConsultantAgent()
        state = MagicMock()
        cb = lambda x: None
        agent.set_shared_state(state)
        agent.set_unrelated_callback(cb)
        assert agent.shared_state is state
        assert agent.unrelated_callback is cb