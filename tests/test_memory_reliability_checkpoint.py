"""检查点只恢复需求草稿；不恢复执行权，所有数据库均来自 tmp_path。"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.constants import StateEnum


class LocalHistory:
    def __init__(self, messages=None) -> None:
        self.messages = list(messages or [])

    def clear(self) -> None:
        self.messages = []


class BookingState:
    """只模拟实际 reset 接口；任何业务写调用都会使测试失败。"""

    def __init__(self, kind: str, shared: LocalHistory) -> None:
        self.kind = kind
        self.session_id = "checkpoint-session"
        self.session_chat_history = shared
        self.chat_history = LocalHistory(["旧的局部提取上下文"])
        self.chats_by_session_id = {self.session_id: self.chat_history}
        self.finished = True
        self.reset_calls = 0
        self.processor = Mock()
        self.processor.handle_complete_trial.side_effect = AssertionError("禁止旧预约")
        self.processor.handle_complete_lesson.side_effect = AssertionError("禁止旧正式课")
        self.change_processor = Mock()
        self.change_processor.handle_change_request.side_effect = AssertionError("禁止调退")
        self.reset()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        defaults = {
            "subject": None,
            "grade_level": None,
            "start_time": None,
            "duration_minutes": "30" if self.kind == "trial" else "60",
            "preferred_teacher_name": None,
            "preferred_style": None,
            "unrelated_escalations": 0,
        }
        if self.kind == "trial":
            defaults = {**defaults, "student_situation": None}
        else:
            defaults = {**defaults, "package_id": None}
        setattr(self, f"{self.kind}_history", defaults)
        self.finished = False
        self.chat_history.clear()


class RecommendationState:
    def __init__(self, shared: LocalHistory) -> None:
        self.session_id = "checkpoint-session"
        self.session_chat_history = shared
        self.history = {self.session_id: self._default_history()}
        self.chats_by_session_id = {}
        self.service = Mock()
        self.service.recommend.side_effect = AssertionError("禁止执行旧推荐")

    @staticmethod
    def _default_history() -> dict:
        return {
            "last_recommendations": [],
            "subject": None,
            "grade_level": None,
            "preferred_style": None,
            "awaiting_confirmation": False,
            "confirmed_teacher": None,
            "recommendation_declined": False,
        }


def make_agent(flow: str = "trial_confirm") -> SimpleNamespace:
    shared = LocalHistory(["家长：数学初中", "顾问：请确认原预约"])
    state = SimpleNamespace(value=flow)
    return SimpleNamespace(
        session_id="checkpoint-session",
        session_chat_history=shared,
        state=state,
        state_manager=SimpleNamespace(state=state),
        trial_agent=BookingState("trial", shared),
        lesson_agent=BookingState("lesson", shared),
        recommendation_agent=RecommendationState(shared),
        payment_service=Mock(),
    )


def checkpoint_row(payload: object, state: str = "trial_confirm",
                   pending_ops: object = None) -> dict:
    return {
        "state": state,
        "payload": json.dumps(payload, ensure_ascii=False),
        "pending_ops": json.dumps(pending_ops),
    }


def populated_agent(flow: str = "trial_confirm") -> SimpleNamespace:
    agent = make_agent(flow)
    safe = {
        "subject": "数学",
        "grade_level": "初中",
        "preferred_style": "温和",
        "preferred_teacher_name": "周敏",
        "student_situation": "基础薄弱，希望提高应用题能力",
        "start_time": "2030-10-03 15:00",
        "duration_minutes": "45",
    }
    unsafe = {
        "teacher_id": 7,
        "confirmed": True,
        "awaiting_confirmation": True,
        "pending_trial": {"teacher_id": 7, "confirmed": True},
        "pending_time_query": "明天下午",
        "awaiting_convert": True,
        "package_id": 3,
        "selected_package": {"id": 3},
    }
    agent.trial_agent.trial_history = {**safe, **unsafe}
    agent.lesson_agent.lesson_history = {**safe, **unsafe}
    recommendation = {
        **safe,
        "last_recommendations": [{"teacher": {"id": 7, "name": "周敏"}}],
        "confirmed_teacher": {"teacher": {"id": 7, "name": "周敏"}},
        "awaiting_confirmation": True,
    }
    agent.recommendation_agent.history = {agent.session_id: recommendation}
    return agent


def assert_no_execution(agent: SimpleNamespace) -> None:
    for child in (agent.trial_agent, agent.lesson_agent):
        assert not child.processor.mock_calls
        assert not child.change_processor.mock_calls
    assert not agent.payment_service.mock_calls
    assert not agent.recommendation_agent.service.mock_calls


def assert_safe_reset(agent: SimpleNamespace) -> None:
    assert agent.state.value == StateEnum.CLASSIFY
    assert agent.state_manager.state.value == StateEnum.CLASSIFY
    for child, kind in ((agent.trial_agent, "trial"), (agent.lesson_agent, "lesson")):
        history = getattr(child, f"{kind}_history")
        assert history["start_time"] is None
        assert history["duration_minutes"] == ("30" if kind == "trial" else "60")
        assert history.get("teacher_id") is None
        assert history.get("package_id") is None
        for key in ("confirmed", "pending_trial", "pending_time_query", "awaiting_convert"):
            assert key not in history
        assert not child.finished
        assert child.chat_history.messages == []
    recommendation = agent.recommendation_agent.history[agent.session_id]
    assert recommendation["last_recommendations"] == []
    assert recommendation["confirmed_teacher"] is None
    assert recommendation["awaiting_confirmation"] is False
    assert_no_execution(agent)


def test_capture_is_json_serializable_allowlisted_and_detached() -> None:
    from services.session_checkpoint import capture_checkpoint

    agent = populated_agent()
    before = copy.deepcopy(agent.trial_agent.trial_history)
    payload = capture_checkpoint(agent)
    assert payload["version"] == 1
    assert payload["source_flow"] == "trial_confirm"
    assert payload["requires_reconfirmation"] is True
    assert set(payload["drafts"]) == {"trial", "lesson", "recommendation"}
    allowed = {"subject", "grade_level", "preferred_style", "preferred_teacher_name",
               "student_situation", "start_time", "duration_minutes"}
    for draft in payload["drafts"].values():
        assert set(draft) <= allowed
        assert draft["subject"] == "数学"
        assert draft["preferred_teacher_name"] == "周敏"
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert agent.trial_agent.trial_history == before
    payload["drafts"]["trial"]["subject"] = "英语"
    assert agent.trial_agent.trial_history["subject"] == "数学"
    assert_no_execution(agent)


@pytest.mark.parametrize("field", ["subject", "grade_level", "preferred_style",
                                   "preferred_teacher_name", "student_situation",
                                   "start_time", "duration_minutes"])
@pytest.mark.parametrize(
    "invalid", [7, True, [], {}, "", "   ", "x" * 10000, "bad\x00value"],
    ids=["number", "boolean", "list", "dict", "empty", "whitespace", "oversized", "control"],
)
def test_capture_rejects_unvalidated_values(field: str, invalid: object) -> None:
    from services.session_checkpoint import capture_checkpoint

    agent = make_agent()
    agent.trial_agent.trial_history = {field: invalid}
    assert field not in capture_checkpoint(agent)["drafts"]["trial"]


@pytest.mark.parametrize("flow", ["trial", "trial_confirm", "trial_convert", "lesson",
                                  "recommendation_confirm", "consultation"])
def test_restore_resets_flow_and_only_refills_safe_requirements(flow: str) -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    source = populated_agent(flow)
    payload = capture_checkpoint(source)
    target = populated_agent("lesson")
    shared = target.session_chat_history
    original_messages = list(shared.messages)
    row = checkpoint_row(payload, flow, [{"operation": "book", "confirmed": True}])
    before_row = copy.deepcopy(row)
    result = restore_checkpoint(target, row)
    assert result["requires_reconfirmation"] is True
    assert "重新确认" in result["notice"]
    assert "时间" in result["notice"]
    assert result["drafts"] == payload["drafts"]
    assert result["drafts"]["trial"]["start_time"] == "2030-10-03 15:00"
    assert_safe_reset(target)
    assert target.trial_agent.trial_history["subject"] == "数学"
    assert target.lesson_agent.lesson_history["preferred_teacher_name"] == "周敏"
    assert target.recommendation_agent.history[target.session_id]["subject"] == "数学"
    assert target.session_chat_history is shared
    assert shared.messages == original_messages
    assert row == before_row
    assert restore_checkpoint(target, row) == result
    assert shared.messages == original_messages


@pytest.mark.parametrize("bad_row", [
    None, "bad", [], {}, {"payload": "not json"},
    {"state": "trial", "payload": "[]", "pending_ops": "[]"},
    {"state": "trial", "payload": '{"version":', "pending_ops": "[]"},
    checkpoint_row({"version": 0, "drafts": {"trial": {"subject": "数学"}}}),
    checkpoint_row({"version": 2, "drafts": {"trial": {"subject": "数学"}}}),
    checkpoint_row({"version": True, "drafts": {"trial": {"subject": "数学"}}}),
    checkpoint_row({"version": 1, "drafts": []}),
    checkpoint_row({"version": 1, "drafts": {"trial": []}}),
    {"state": "trial", "payload": '{"version":1,"drafts":{}}', "pending_ops": "bad"},
])
def test_malformed_or_legacy_checkpoints_safely_invalidate(bad_row: object) -> None:
    from services.session_checkpoint import restore_checkpoint

    agent = populated_agent()
    messages = list(agent.session_chat_history.messages)
    result = restore_checkpoint(agent, bad_row)
    assert_safe_reset(agent)
    assert result["drafts"] == {"trial": {}, "lesson": {}, "recommendation": {}}
    assert result["requires_reconfirmation"] is True
    assert "重新确认" in result["notice"]
    assert agent.trial_agent.trial_history["subject"] is None
    assert agent.session_chat_history.messages == messages


def test_restore_revalidates_payload_instead_of_trusting_saved_fields() -> None:
    from services.session_checkpoint import restore_checkpoint

    payload = {
        "version": 1,
        "source_flow": "trial_confirm",
        "requires_reconfirmation": False,
        "drafts": {
            "trial": {"subject": "数学", "grade_level": ["初中"], "teacher_id": 7,
                      "start_time": "2030-10-03 15:00", "duration_minutes": "45",
                      "preferred_teacher_name": "x" * 10000, "confirmed": True},
            "lesson": {"package_id": 3, "pending_trial": {"teacher_id": 7}},
            "recommendation": {"last_recommendations": [{"id": 7}]},
        },
    }
    agent = populated_agent()
    result = restore_checkpoint(agent, checkpoint_row(payload))
    assert result["drafts"]["trial"] == {
        "subject": "数学", "start_time": "2030-10-03 15:00", "duration_minutes": "45",
    }
    assert result["drafts"]["lesson"] == {}
    assert result["drafts"]["recommendation"] == {}
    assert result["requires_reconfirmation"] is True
    assert_safe_reset(agent)


def test_empty_classify_checkpoint_needs_no_business_reconfirmation() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = make_agent(StateEnum.CLASSIFY)
    payload = capture_checkpoint(agent)
    assert payload["requires_reconfirmation"] is False
    result = restore_checkpoint(agent, checkpoint_row(payload, "classify", []))
    assert result["requires_reconfirmation"] is False
    assert result["notice"] == ""
    assert_safe_reset(agent)


def test_empty_saved_checkpoint_cannot_hide_current_pending_operation() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    empty = capture_checkpoint(make_agent(StateEnum.CLASSIFY))
    target = populated_agent("trial_confirm")
    result = restore_checkpoint(target, checkpoint_row(empty, "classify", []))
    assert result["requires_reconfirmation"] is True
    assert "重新确认" in result["notice"]
    assert_safe_reset(target)


@pytest.mark.parametrize("failing_kind", ["trial", "lesson", "recommendation"])
def test_reset_failure_invalidates_restore_but_clears_other_operations(
    failing_kind: str, monkeypatch, caplog,
) -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    row = checkpoint_row(capture_checkpoint(agent))
    child = getattr(agent, f"{failing_kind}_agent")
    method = "_default_history" if failing_kind == "recommendation" else "reset"
    monkeypatch.setattr(child, method, Mock(side_effect=RuntimeError("reset failed")))
    result = restore_checkpoint(agent, row)
    assert result["requires_reconfirmation"] is True
    assert result["drafts"] == {"trial": {}, "lesson": {}, "recommendation": {}}
    assert "重新确认" in result["notice"]
    assert "checkpoint" in caplog.text.lower()
    assert_safe_reset(agent)


def test_reset_does_not_clear_even_aliased_session_history() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    row = checkpoint_row(capture_checkpoint(agent))
    shared = agent.session_chat_history
    messages = list(shared.messages)
    agent.trial_agent.chat_history = shared
    agent.trial_agent.chats_by_session_id = {agent.session_id: shared}
    restore_checkpoint(agent, row)
    assert shared.messages == messages
    assert agent.trial_agent.chat_history is not shared
    assert_safe_reset(agent)


def test_reset_protects_direct_shared_messages_list_alias() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    row = checkpoint_row(capture_checkpoint(agent))
    shared = agent.session_chat_history
    messages = list(shared.messages)
    agent.trial_agent.chat_history = shared.messages
    restore_checkpoint(agent, row)
    assert shared.messages == messages
    assert agent.trial_agent.chat_history is not shared.messages
    assert agent.trial_agent.chat_history == []
    assert agent.trial_agent.trial_history.get("pending_trial") is None
    assert_no_execution(agent)


def test_only_current_recommendation_session_is_reset() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    other = {"subject": "英语", "last_recommendations": [{"teacher": {"id": 9}}]}
    agent.recommendation_agent.history = {
        **agent.recommendation_agent.history, "other-session": other,
    }
    restore_checkpoint(agent, checkpoint_row(capture_checkpoint(agent)))
    assert agent.recommendation_agent.history["other-session"] == other
    assert_safe_reset(agent)


def test_missing_optional_children_are_supported() -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    agent = SimpleNamespace(state=SimpleNamespace(value=StateEnum.CLASSIFY))
    payload = capture_checkpoint(agent)
    assert payload["drafts"] == {"trial": {}, "lesson": {}, "recommendation": {}}
    result = restore_checkpoint(agent, checkpoint_row(payload, "classify", []))
    assert result["requires_reconfirmation"] is False
    assert agent.state.value == StateEnum.CLASSIFY


def test_real_temp_db_rebuild_preserves_draft_but_never_executes(tmp_path) -> None:
    from config.identity import Identity
    from services.conversation_service import ConversationService
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    db_url = f"sqlite:///{tmp_path / 'checkpoint.db'}"
    identity = Identity(user_id="checkpoint-owner", session_id="checkpoint-session")
    source = populated_agent()
    payload = capture_checkpoint(source)
    service1 = ConversationService(db_path=db_url)
    try:
        service1.record_human(identity, "数学初中，周敏老师，明天下午")
        service1.record_ai(identity, "请确认旧试讲草稿")
        service1.save_checkpoint(identity, payload["source_flow"], payload,
                                 [{"op": "book_trial", "teacher_id": 7, "confirmed": True}])
    finally:
        service1.close()

    service2 = ConversationService(db_path=db_url)
    try:
        loaded = service2.load_checkpoint(identity)
        assert json.loads(loaded["payload"]) == payload
        target = populated_agent("lesson")
        target.session_chat_history = service2.restore_history(identity)
        history_before = list(target.session_chat_history.messages)
        before_count = service2.message_count(identity)
        result = restore_checkpoint(target, loaded)
        assert result["requires_reconfirmation"] is True
        assert result["drafts"]["trial"]["start_time"] == "2030-10-03 15:00"
        assert_safe_reset(target)
        assert target.trial_agent.trial_history["subject"] == "数学"
        assert target.session_chat_history.messages == history_before
        assert service2.message_count(identity) == before_count == 2
        assert service2.load_checkpoint(identity) == loaded
        assert restore_checkpoint(target, loaded) == result
        assert_no_execution(source)
    finally:
        service2.close()
