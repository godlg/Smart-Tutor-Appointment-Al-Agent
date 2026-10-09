"""异常清理与共享消息别名的检查点安全回归；不调用业务服务。"""

import logging

import pytest

from config.constants import StateEnum
from test_memory_reliability_checkpoint import (
    assert_no_execution,
    assert_safe_reset,
    checkpoint_row,
    make_agent,
    populated_agent,
)


class InPlaceHistory:
    def __init__(self, messages: list) -> None:
        self.messages = messages

    def clear(self) -> None:
        self.messages.clear()


@pytest.mark.parametrize("kind", ["trial", "lesson", "recommendation"])
def test_current_pending_requires_notice_even_with_classify_empty_checkpoint(kind: str) -> None:
    from services.session_checkpoint import capture_checkpoint, restore_checkpoint

    empty = capture_checkpoint(make_agent(StateEnum.CLASSIFY))
    agent = make_agent(StateEnum.CLASSIFY)
    child = getattr(agent, f"{kind}_agent")
    if kind == "recommendation":
        child.history = {agent.session_id: {"awaiting_confirmation": True}}
    else:
        setattr(child, f"{kind}_history", {"pending_trial": {"teacher_id": 7}})

    result = restore_checkpoint(agent, checkpoint_row(empty, "classify", []))

    assert result["requires_reconfirmation"] is True
    assert "重新确认" in result["notice"]
    assert_safe_reset(agent)


@pytest.mark.parametrize("kind", ["trial", "lesson"])
@pytest.mark.parametrize("alias", ["shared", "wrapper", "messages"])
@pytest.mark.parametrize("fails_after_clear", [False, True])
@pytest.mark.parametrize("shared_as_list", [False, True])
def test_shared_alias_is_protected_even_after_partial_reset_failure(
    kind: str, alias: str, fails_after_clear: bool, shared_as_list: bool, monkeypatch, caplog,
) -> None:
    from services.session_checkpoint import INVALID_NOTICE, capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    row = checkpoint_row(capture_checkpoint(agent))
    shared = agent.session_chat_history
    messages = shared.messages
    original_messages = list(messages)
    if shared_as_list:
        shared = messages
        agent.session_chat_history = shared
        for kind_child in (agent.trial_agent, agent.lesson_agent, agent.recommendation_agent):
            kind_child.session_chat_history = shared
    child = getattr(agent, f"{kind}_agent")
    local = {"shared": shared, "wrapper": InPlaceHistory(messages), "messages": messages}[alias]
    other = InPlaceHistory(["无关会话"])
    child.chat_history = local
    child.chats_by_session_id = {agent.session_id: local, "other-session": other}
    if fails_after_clear:
        def partial_reset() -> None:
            child.chat_history.clear()
            setattr(child, f"{kind}_history", {"pending_trial": {"teacher_id": 99}})
            raise RuntimeError("partial reset failure")
        monkeypatch.setattr(child, "reset", partial_reset)

    result = restore_checkpoint(agent, row)

    assert agent.session_chat_history is shared
    assert (shared if shared_as_list else shared.messages) is messages
    assert messages == original_messages
    assert child.chat_history is not local
    assert child.chats_by_session_id[agent.session_id] is child.chat_history
    assert child.chats_by_session_id["other-session"] is other
    local_messages = (child.chat_history if isinstance(child.chat_history, list)
                      else child.chat_history.messages)
    assert local_messages == []
    assert getattr(child, f"{kind}_history").get("pending_trial") is None
    assert_no_execution(agent)
    if fails_after_clear:
        assert result["notice"] == INVALID_NOTICE
        assert result["drafts"] == {"trial": {}, "lesson": {}, "recommendation": {}}
        assert all(draft.get("subject") is None for draft in (
            agent.trial_agent.trial_history,
            agent.lesson_agent.lesson_history,
            agent.recommendation_agent.history[agent.session_id],
        ))
        assert any(record.levelno >= logging.WARNING and "checkpoint" in record.name
                   and kind in record.getMessage() for record in caplog.records)


def test_all_children_reset_failures_still_clear_pending_and_other_session_is_untouched(
    monkeypatch, caplog,
) -> None:
    from services.session_checkpoint import INVALID_NOTICE, capture_checkpoint, restore_checkpoint

    agent = populated_agent()
    row = checkpoint_row(capture_checkpoint(agent))
    other = {"subject": "英语", "awaiting_confirmation": True}
    recommendation = agent.recommendation_agent
    recommendation.history = {**recommendation.history, "other-session": other}
    def fail_reset() -> None:
        raise RuntimeError("reset unavailable")
    for child in (agent.trial_agent, agent.lesson_agent):
        monkeypatch.setattr(child, "reset", fail_reset)
    monkeypatch.setattr(recommendation, "_default_history", fail_reset)

    result = restore_checkpoint(agent, row)

    assert result["requires_reconfirmation"] is True
    assert result["notice"] == INVALID_NOTICE
    assert result["drafts"] == {"trial": {}, "lesson": {}, "recommendation": {}}
    assert_safe_reset(agent)
    assert recommendation.history["other-session"] is other
    assert all(draft.get("subject") is None for draft in (
        agent.trial_agent.trial_history, agent.lesson_agent.lesson_history,
        recommendation.history[agent.session_id],
    ))
    for kind in ("trial", "lesson", "recommendation"):
        assert any(record.levelno >= logging.WARNING and "checkpoint" in record.name
                   and kind in record.getMessage() for record in caplog.records)
