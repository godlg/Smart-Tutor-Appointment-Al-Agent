"""统一历史的真实分类/业务调用；仅模型和业务服务外边界 fake。"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.context import (
    ContextManager, ConversationContext, StructuredOutputContext, SystemPromptContext,
)
from agents.recommendation_agent import RecommendationAgent, RecommendationInputParser
from agents.task_classification.task_classifier import TaskClassifier
from agents.task_classification_agent import TaskClassificationAgent
from agents.trial.trial_agent import TrialAgent
from agents.trial.trial_processor import TrialInputParser


class ManagedHistory(InMemoryChatMessageHistory):
    entry_managed: bool = True


def _model(response):
    prompts = []

    def invoke(value):
        if hasattr(value, "to_string"):
            text = value.to_string()
        else:
            text = "\n".join(message["content"] for message in value)
        prompts.append(text)
        return AIMessage(content=response)

    return RunnableLambda(invoke), prompts


def _manager(history, session_id="routing-session"):
    def provider(sid):
        assert sid == session_id
        return history

    manager = ContextManager([
        SystemPromptContext(), ConversationContext(history_provider=provider),
        StructuredOutputContext(),
    ])
    manager.max_tokens = 6000
    return manager


async def _collect(stream):
    return "".join([part async for part in stream])


def _history(managed=True):
    history = ManagedHistory() if managed else InMemoryChatMessageHistory()
    history.add_user_message("孩子初二数学，需要温和的老师")
    history.add_ai_message("已为您确认 周敏 老师(学科: 数学, 年级: 初中, 老师ID: 7)")
    return history


@pytest.fixture
def offline_agents(monkeypatch):
    from services import lesson_service, student_profile_service
    monkeypatch.setattr(lesson_service, "LessonService", MagicMock())
    monkeypatch.setattr(student_profile_service, "StudentProfileService", MagicMock())


def test_actual_controller_classification_keeps_shared_history_and_intent_contract(monkeypatch):
    from agents import task_classification_agent
    history = _history()
    history.add_user_message("试讲")
    model, prompts = _model("trial")
    monkeypatch.setattr(task_classification_agent, "create_chat_model", lambda **kw: model)
    handoff = MagicMock()
    handoff.is_handed_off.return_value = False
    handoff.match_keyword.return_value = None
    agent = TaskClassificationAgent(
        session_id="routing-session", session_chat_history=history,
        context_manager=_manager(history), handoff_service=handoff,
    )

    assert asyncio.run(agent.task_classifier.classify_task("试讲")) == "trial"
    prompt = prompts[-1]
    assert "孩子初二数学" in prompt
    assert "周敏" in prompt
    assert "K12" in prompt
    assert "请只输出上述英文类别之一" in prompt
    assert "recommendation_confirm" in prompt
    assert "本次输入" in prompt or "用户问题：试讲" in prompt
    assert len(history.messages) == 3


@pytest.mark.parametrize("agent_class", [TaskClassificationAgent, TrialAgent])
def test_internal_context_builder_reads_injected_shared_history(agent_class, monkeypatch, offline_agents):
    from agents import task_classification_agent
    from agents.trial import trial_agent
    model, _ = _model("trial")
    for module in (task_classification_agent, trial_agent):
        monkeypatch.setattr(module, "create_chat_model", lambda **kw: model)
    history = _history()
    kwargs = {"session_id": "routing-session", "session_chat_history": history}
    if agent_class is TaskClassificationAgent:
        kwargs["handoff_service"] = MagicMock()
    agent = agent_class(**kwargs)

    prompt = agent._build_context_manager().build(task="试讲")
    assert "孩子初二数学" in prompt
    assert "周敏" in prompt


def test_actual_trial_agent_reads_only_unified_history_keeps_json_and_does_not_shared_write(monkeypatch, offline_agents):
    from agents.trial import trial_agent
    data = {
        "subject": "数学", "grade_level": "初中", "start_time": "未知",
        "duration_minutes": "30", "info_complete": False, "unrelated": False,
        "missing_info": ["start_time"],
    }
    model, prompts = _model(json.dumps(data, ensure_ascii=False))
    monkeypatch.setattr(trial_agent, "create_chat_model", lambda **kw: model)
    history = _history()
    history.add_user_message("试讲")
    agent = TrialAgent(
        session_id="routing-session", session_chat_history=history,
        context_manager=_manager(history),
    )
    agent.chat_history.add_ai_message("LOCAL_EXTRACTION_JSON_NOT_USER_VISIBLE")

    output = asyncio.run(_collect(agent.run_stream("试讲")))

    assert "时间" in output or "start_time" in output
    prompt = prompts[-1]
    assert "孩子初二数学" in prompt
    assert "周敏" in prompt
    assert "只输出纯 JSON" in prompt
    assert '"start_time"' in prompt and '"subject"' in prompt
    assert "LOCAL_EXTRACTION_JSON_NOT_USER_VISIBLE" not in prompt
    assert agent.trial_history["teacher_id"] == 7
    assert len(history.messages) == 3
    # 局部提取历史仍兼容，但不成为共享的可见AI回复。
    assert any(message.content.startswith("{") for message in agent.chat_history.messages)


@pytest.mark.parametrize("agent_class", [TrialAgent, RecommendationAgent])
@pytest.mark.parametrize("managed", [True, False])
def test_shared_ai_write_ownership_never_deduplicates_by_content(agent_class, managed, monkeypatch, offline_agents):
    from agents.trial import trial_agent
    from agents import recommendation_agent
    model, _ = _model("{}")
    for module in (trial_agent, recommendation_agent):
        monkeypatch.setattr(module, "create_chat_model", lambda **kw: model)
    history = _history(managed=managed)
    agent = agent_class(session_id="routing-session", session_chat_history=history)
    original = list(history.messages)

    agent._record_shared_ai("重复回复")
    agent._record_shared_ai("重复回复")

    if managed:
        assert history.messages == original
    else:
        assert [message.content for message in history.messages[-2:]] == ["重复回复", "重复回复"]


def test_actual_recommendation_extract_uses_session_history_and_json_instruction(monkeypatch):
    from agents import recommendation_agent
    model, prompts = _model(json.dumps({
        "subject": "未知", "grade_level": "未知", "info_complete": False,
    }))
    monkeypatch.setattr(recommendation_agent, "create_chat_model", lambda **kw: model)
    history = _history()
    history.add_user_message("再推荐一位")
    agent = RecommendationAgent(
        session_id="routing-session", session_chat_history=history,
        context_manager=_manager(history),
    )

    output = asyncio.run(_collect(agent.run_stream("再推荐一位")))

    assert "学科" in output
    prompt = prompts[-1]
    assert "孩子初二数学" in prompt
    assert "周敏" in prompt
    assert "只输出纯 JSON" in prompt
    assert '"subject"' in prompt and '"grade_level"' in prompt
    assert len(history.messages) == 3


def test_parser_never_writes_extraction_json_to_entry_managed_history():
    history = _history()
    history.add_user_message("试讲")
    manager = _manager(history)
    manager.session_id = "routing-session"
    model, prompts = _model('{"subject":"数学","unrelated":false}')
    parser = TrialInputParser(model, context_manager=manager)
    original = list(history.messages)

    data = parser.parse_data("".join(parser.parse_stream("试讲", history)))

    assert data["subject"] == "数学"
    assert history.messages == original
    assert prompts[-1].count("孩子初二数学") == 1


class _BusinessBoundary:
    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append((user_input, intent))
        yield "[REPLY]DEFAULT_STUDENT_BUSINESS"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def consult(self, task):
        self.calls.append(task)
        return "DEFAULT_STUDENT_BUSINESS"

    async def consult_stream(self, task):
        self.calls.append(task)
        yield "[REPLY]DEFAULT_STUDENT_BUSINESS"


def _business_controller(monkeypatch, category):
    from agents import task_classification_agent
    model, _ = _model(category)
    monkeypatch.setattr(task_classification_agent, "create_chat_model", lambda **kw: model)
    business = _BusinessBoundary()
    handoff = MagicMock()
    handoff.is_handed_off.return_value = False
    handoff.match_keyword.return_value = None
    agent = TaskClassificationAgent(
        consultant_agent=business, trial_agent=business, lesson_agent=business,
        session_id="routing-session", handoff_service=handoff,
    )
    return agent, business


def _call_with_identity(agent, identity, mode):
    from config.identity import set_current_identity, reset_current_identity
    token = set_current_identity(identity)
    try:
        if mode == "sync":
            return asyncio.run(agent.classify_task("确认预约或查询孩子进度"))
        return asyncio.run(_collect(agent.classify_task_stream("确认预约或查询孩子进度")))
    finally:
        reset_current_identity(token)


@pytest.mark.parametrize("mode", ["sync", "stream"])
@pytest.mark.parametrize("category", ["trial", "lesson", "reschedule", "cancel", "progress"])
def test_anonymous_classified_business_request_requires_verification(monkeypatch, mode, category):
    from config.identity import Identity
    agent, business = _business_controller(monkeypatch, category)

    output = _call_with_identity(agent, Identity("anon_parent", student_id="1"), mode)

    assert "认证" in output or "验证" in output
    assert business.calls == []
    assert "DEFAULT_STUDENT_BUSINESS" not in output


@pytest.mark.parametrize("mode", ["sync", "stream"])
@pytest.mark.parametrize("state", ["trial", "trial_confirm", "trial_convert", "lesson"])
def test_anonymous_business_continuation_cannot_bypass_verification(monkeypatch, mode, state):
    from config.identity import Identity
    from agents.task_classification.state_manager import FlowState
    agent, business = _business_controller(monkeypatch, "consultation")
    agent.state.value = FlowState(state)

    output = _call_with_identity(agent, Identity("anon_parent", student_id="1"), mode)

    assert "认证" in output or "验证" in output
    assert business.calls == []


@pytest.mark.parametrize("identity_kind", ["unset", "legacy"])
def test_legacy_business_route_remains_available(monkeypatch, identity_kind):
    from config.identity import legacy_identity
    agent, business = _business_controller(monkeypatch, "trial")
    identity = legacy_identity("routing-session") if identity_kind == "legacy" else None

    output = _call_with_identity(agent, identity, "stream")

    assert "DEFAULT_STUDENT_BUSINESS" in output
    assert len(business.calls) == 1


def test_anonymous_public_recommendation_does_not_initialize_default_student(monkeypatch):
    from config.identity import Identity, set_current_identity, reset_current_identity
    from agents import recommendation_agent
    model, _ = _model(json.dumps({
        "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
        "info_complete": True,
    }, ensure_ascii=False))
    monkeypatch.setattr(recommendation_agent, "create_chat_model", lambda **kw: model)
    agent = RecommendationAgent(session_id="routing-session")
    student = MagicMock()
    student.initialize_default_student.side_effect = AssertionError("anonymous must not read default student")
    agent._student_service = student
    recommendation = MagicMock()
    recommendation.recommend_teachers.return_value = []
    agent._recommendation_service = recommendation
    token = set_current_identity(Identity("anon_parent", student_id="1"))
    try:
        output = asyncio.run(_collect(agent.run_stream("初二数学推荐温和老师")))
    finally:
        reset_current_identity(token)

    assert "没有找到" in output
    student.initialize_default_student.assert_not_called()
    recommendation.recommend_teachers.assert_called_once_with(
        student={"grade_level": "初中", "preferred_style": "温和"}, subject="数学", top_k=3,
    )


def test_anonymous_recommendation_confirms_public_teacher_but_not_booking(monkeypatch):
    from config.identity import Identity, set_current_identity, reset_current_identity
    from agents import recommendation_agent
    from agents.task_classification.state_manager import FlowState, SharedState
    model, prompts = _model("{}")
    monkeypatch.setattr(recommendation_agent, "create_chat_model", lambda **kw: model)
    agent = RecommendationAgent(session_id="routing-session")
    state = SharedState()
    state.value = FlowState.RECOMMENDATION_CONFIRM
    agent.set_shared_state(state)
    agent.history[agent.session_id] = {
        **agent._default_history(), "subject": "数学", "grade_level": "初中",
        "awaiting_confirmation": True,
        "confirmed_teacher": {"teacher": {"id": 7, "name": "周敏"}},
    }
    token = set_current_identity(Identity("anon_parent", student_id="1"))
    try:
        output = asyncio.run(_collect(agent.run_stream("是")))
    finally:
        reset_current_identity(token)

    assert "周敏" in output
    assert "认证" in output or "验证" in output
    assert state.value == FlowState.CLASSIFY
    assert prompts == []


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_classification_budget_rejection_is_explicit_and_not_a_kickback(monkeypatch, mode):
    agent, business = _business_controller(monkeypatch, "consultation")
    model, prompts = _model("consultation")
    agent.task_classifier.llm = model
    manager = ContextManager([SystemPromptContext()])
    manager.max_tokens = 1
    agent.task_classifier.context_manager = manager

    output = _call_with_identity(agent, None, mode)

    assert "预算" in output and "缩短" in output
    assert output.startswith("[ERROR]")
    assert prompts == []
    assert business.calls == []
    agent.handoff_service.record_kickback.assert_not_called()
    agent.handoff_service.trigger_handoff.assert_not_called()


def test_recommendation_budget_rejection_is_not_a_missing_field_question(monkeypatch):
    from agents import recommendation_agent
    model, prompts = _model("{}")
    monkeypatch.setattr(recommendation_agent, "create_chat_model", lambda **kw: model)
    manager = ContextManager([SystemPromptContext()])
    manager.max_tokens = 1
    agent = RecommendationAgent(session_id="routing-session", context_manager=manager)

    output = asyncio.run(_collect(agent.run_stream("孩子初二数学需要温和的老师")))

    assert "预算" in output and "缩短" in output
    assert "请问需要辅导哪个学科" not in output
    assert prompts == []


def test_legacy_classifier_trial_and_recommendation_optional_paths_are_compatible():
    classifier_model, classifier_prompts = _model("trial")
    classifier = TaskClassifier(classifier_model)
    assert asyncio.run(classifier.classify_task("试讲")) == "trial"
    assert "请只输出上述英文类别之一" in classifier_prompts[-1]

    parser_model, parser_prompts = _model('{"subject":"数学","info_complete":"false"}')
    parser = TrialInputParser(parser_model)
    history = InMemoryChatMessageHistory()
    result = parser.parse_data("".join(parser.parse_stream("试讲", history, context="上游确认周敏")))
    assert result["subject"] == "数学" and result["info_complete"] is False
    assert [message.type for message in history.messages] == ["human", "ai"]
    assert "上游确认周敏" in parser_prompts[-1]

    recommendation = RecommendationInputParser(parser_model)
    assert recommendation.extract("推荐数学老师")["subject"] == "数学"
