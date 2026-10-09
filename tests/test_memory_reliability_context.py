"""Context reliability: offline RED/GREEN coverage for hard budgets and rollups."""
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.context.base import BaseContext
from agents.context.contexts.conversation import ConversationContext
from agents.context.manager import ContextManager
from agents.context.types import ContextBudgetExceeded, estimate_token_count
from config.identity import Identity


class CountingContext(BaseContext):
    def __init__(self, name, text, priority=20):
        self.name = name
        self.text = text
        self.priority = priority
        self.calls = 0

    def build(self, manager):
        self.calls += 1
        return self.text


def history_context(rounds=12, session="budget-session", filler=""):
    conv = ConversationContext(keep_rounds=4)
    for i in range(rounds):
        conv.get_history(session).add_messages([
            HumanMessage(content=f"旧问题{i:02d} {filler}"),
            AIMessage(content=f"旧回答{i:02d} {filler}"),
        ])
    return conv


def test_hard_budget_rejects_required_text_instead_of_cutting_question():
    manager = ContextManager([CountingContext("system_prompt", "必要系统" * 30, 0)])
    with pytest.raises(ContextBudgetExceeded) as error:
        manager.build(task="必须完整保留的问题", max_tokens=10)
    assert error.value.used > error.value.budget


def test_hard_budget_trims_rag_even_without_conversation():
    manager = ContextManager([CountingContext("system_prompt", "SYSTEM", 0)])
    result = manager.build(task="问题", extra_contexts={"knowledge": "数学" * 2000}, max_tokens=100)
    assert estimate_token_count(result) <= 100
    assert "SYSTEM" in result and "用户问题：问题" in result


def test_each_context_builds_once_even_during_rollup_compression_and_report():
    memory = CountingContext("memory", "记忆" * 300)
    conv = history_context(filler="历史" * 20)
    manager = ContextManager([conv, memory], rollup_enabled=True,
                             rollup_persister=lambda **kwargs: True)
    manager.build(task="问题", session_id="budget-session", max_tokens=100)
    manager.usage_report()
    manager.usage_report()
    assert memory.calls == 1


def test_usage_report_before_first_build_does_not_recall():
    memory = CountingContext("memory", "摘要")
    manager = ContextManager([memory])
    manager.usage_report()
    assert memory.calls == 0


def test_compressed_history_never_resurrects_rolled_messages():
    conv = history_context(rounds=6)
    conv.mark_rolled_up("budget-session", 10)
    result = conv.build_compressed(SimpleNamespace(session_id="budget-session"))
    assert "旧问题02" not in result
    assert "旧问题05" in result


@pytest.mark.parametrize("failure", [False, RuntimeError("storage unavailable")])
def test_failed_rollup_keeps_watermark_and_original_text(failure, caplog):
    conv = history_context()

    def persist(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    manager = ContextManager([conv], rollup_enabled=True,
                             rollup_window_ratio=0.01, rollup_persister=persist)
    with caplog.at_level(logging.WARNING):
        result = manager.build(session_id="budget-session", max_tokens=10000)
    assert conv._rolled_up_count("budget-session") == 0
    assert not manager.last_rollup_triggered
    assert "旧问题00" in result
    assert "滚动摘要" in caplog.text or "rollup" in caplog.text.lower()


def test_rollup_progresses_from_current_watermark_three_times_and_then_stops():
    conv = history_context(rounds=10)
    saved = []

    def legacy_persister(user_id, summary_text, session_id=None):
        saved.append(summary_text)
        # Legacy callback's None return is success, preserving its old signature.

    manager = ContextManager([conv], rollup_enabled=True, rollup_rounds=2,
                             rollup_window_ratio=0.01, rollup_persister=legacy_persister)
    watermarks = []
    for _ in range(3):
        manager.build(session_id="budget-session", max_tokens=10000)
        watermarks.append(conv._rolled_up_count("budget-session"))
    assert watermarks == [4, 8, 12]
    manager.build(session_id="budget-session", max_tokens=10000)
    assert len(saved) == 3
    assert not manager.last_rollup_triggered
    assert "旧问题00" in saved[0]
    assert "旧问题02" in saved[1]
    assert "旧问题04" in saved[2]
    result = manager.build(session_id="budget-session", max_tokens=10000)
    assert "旧问题06" in result and "旧问题09" in result


def test_injected_counter_output_reserve_and_default_manager_budget():
    manager = ContextManager([CountingContext("system_prompt", "SYSTEM", 0)],
                             token_counter=len, output_tokens=20)
    manager.max_tokens = 120
    result = manager.build(task="数学", extra_contexts={"knowledge": "K" * 1000})
    assert len(result) <= 100
    assert manager.estimate_tokens() == len(result)
    assert "SYSTEM" in result and "用户问题：数学" in result
    assert len(manager.build(task="数学", max_tokens=150)) <= 130


@pytest.mark.parametrize("name,priority", [("system_prompt", 0), ("tools", 30),
                                           ("structured_output", 40)])
def test_required_blocks_and_extra_business_instructions_are_never_truncated(name, priority):
    business = '{"teacher_id": 77, "must_confirm": true}'
    manager = ContextManager([CountingContext(name, business, priority)], token_counter=len)
    with pytest.raises(ContextBudgetExceeded):
        manager.build(task="问题", max_tokens=20)
    with pytest.raises(ContextBudgetExceeded):
        ContextManager(token_counter=len).build(task="问题", max_tokens=20,
                                                extra_contexts={"instructions": business})


def test_required_failure_propagates_but_optional_failure_degrades(caplog):
    class Broken(CountingContext):
        def build(self, manager):
            raise RuntimeError("failed context")

    with pytest.raises(RuntimeError):
        ContextManager([Broken("system_prompt", "", 0)]).build(task="问题")
    with caplog.at_level(logging.WARNING):
        result = ContextManager([Broken("memory", "")]).build(task="问题")
    assert "用户问题：问题" in result
    assert "memory" in caplog.text


def test_long_single_recent_message_and_multiple_optional_blocks_are_bounded():
    conv = history_context(rounds=1, filler="超长历史" * 5000)
    before = list(conv.get_history("budget-session").messages)
    manager = ContextManager([conv, CountingContext("memory", "记忆" * 5000),
                              CountingContext("system_prompt", "必要系统", 0)],
                             token_counter=len, output_tokens=10)
    result = manager.build(task="当前问题完整", session_id="budget-session", max_tokens=230,
                           extra_contexts={"knowledge": "知识" * 5000,
                                           "extract_instruction": '{"confirm": true}'})
    assert len(result) <= 220
    assert "必要系统" in result and "当前问题完整" in result
    assert '{"confirm": true}' in result
    assert conv.get_history("budget-session").messages == before


def test_budget_exact_fit_and_question_too_large():
    manager = ContextManager(token_counter=len, output_tokens=5)
    unbounded = manager.build(task="问题")
    assert manager.build(task="问题", max_tokens=len(unbounded) + 5) == unbounded
    with pytest.raises(ContextBudgetExceeded):
        manager.build(task="问题", max_tokens=len(unbounded) + 4)
    with pytest.raises(ContextBudgetExceeded):
        manager.build(task="大问题" * 100, max_tokens=100)


def test_extractive_summary_is_bounded_and_prioritizes_confirmed_constraints():
    conv = ConversationContext()
    history = conv.get_history("summary-session")
    history.add_messages([HumanMessage(content="闲聊" * 2000), AIMessage(content="好的" * 1000),
                          HumanMessage(content="确认：六年级数学，王老师，周三19:00，只能线上，待办预约试讲"),
                          AIMessage(content="已经记录，等待家长确认")])
    for _ in range(4):
        history.add_messages([HumanMessage(content="最近问题"), AIMessage(content="最近回答")])
    saved = []
    manager = ContextManager([conv], rollup_enabled=True, rollup_window_ratio=0.01,
                             rollup_rounds=2, summary_max_chars=120,
                             rollup_persister=lambda **kwargs: saved.append(kwargs["summary_text"]))
    result = manager.build(session_id="summary-session", max_tokens=20000)
    assert len(saved[0]) <= 120
    for fact in ["六年级", "数学", "王老师", "周三19:00", "只能线上", "待办"]:
        assert fact in saved[0]
        assert fact in result  # Explicit summary, without memory/embedding context.
    assert "extractive" in saved[0].lower() or "摘录" in saved[0]


def test_injected_summarizer_is_called_once_and_its_output_is_bounded():
    conv = history_context(rounds=6, filler="触发摘要" * 10)
    calls = []

    def summarize(messages, max_chars):
        calls.append((messages, max_chars))
        return "老师确认" * 1000

    saved = []
    manager = ContextManager([conv], rollup_enabled=True, rollup_window_ratio=0.01,
                             rollup_rounds=2, summarizer=summarize, summary_max_chars=80,
                             rollup_persister=lambda **kwargs: saved.append(kwargs["summary_text"]))
    manager.build(session_id="budget-session", max_tokens=10000)
    assert len(calls) == 1 and calls[0][1] == 80
    assert len(saved[0]) <= 80


@pytest.fixture
def durable_conversation(tmp_path):
    from services.conversation_service import ConversationService
    service = ConversationService(db_path=f"sqlite:///{tmp_path / 'context.db'}")
    identity = Identity(user_id="anon_context", student_id="student-7", session_id="durable-context")
    for i in range(10):
        service.record_human(identity, f"原文用户{i:02d}")
        service.record_ai(identity, f"原文顾问{i:02d}")
    yield service, identity, f"sqlite:///{tmp_path / 'context.db'}"
    service.close()


def test_persistent_rollup_restores_watermark_injects_text_and_uses_unique_ranges(durable_conversation):
    from services.conversation_service import ConversationService
    service, identity, db_path = durable_conversation
    conv = ConversationContext(history_provider=service.get_history)
    manager = ContextManager([conv], identity=identity, conversation_service=service,
                             rollup_enabled=True, rollup_rounds=2, rollup_window_ratio=0.01)
    manager.build(max_tokens=10000)
    assert service.summary_watermark(identity) == 4
    restored = ConversationService(db_path=db_path)
    try:
        restored.restore_history(identity)
        new_conv = ConversationContext(history_provider=restored.get_history)
        new_manager = ContextManager([new_conv], identity=identity, conversation_service=restored,
                                     rollup_enabled=True, rollup_rounds=2, rollup_window_ratio=0.01)
        for _ in range(2):
            new_manager.build(max_tokens=10000)
        summaries = restored.get_summaries(identity)
        assert [(s["start_seq"], s["end_seq"]) for s in summaries] == [(1, 4), (5, 8), (9, 12)]
        assert new_conv._rolled_up_count(identity.session_id) == 12
        result = new_manager.build(max_tokens=10000)
        assert len(restored.get_summaries(identity)) == 3
        assert "近期滚动摘要" in result
        assert all(s["content"] in result for s in summaries)
        assert "原文用户09" in result
        # Archived source is present only through the explicit bounded summary block.
        assert "原文用户00" not in result.split("【近期滚动摘要】")[0]
    finally:
        restored.close()


def test_persistent_save_false_does_not_advance_context_watermark(durable_conversation, monkeypatch):
    service, identity, _ = durable_conversation
    conv = ConversationContext(history_provider=service.get_history)
    monkeypatch.setattr(service, "save_rollup", lambda *args: False)
    manager = ContextManager([conv], identity=identity, conversation_service=service,
                             rollup_enabled=True, rollup_rounds=2, rollup_window_ratio=0.01)
    result = manager.build(max_tokens=10000)
    assert conv._rolled_up_count(identity.session_id) == 0
    assert "原文用户00" in result
    assert not manager.last_rollup_triggered


def test_summary_text_has_own_budget_and_needs_no_memory_context(durable_conversation):
    service, identity, _ = durable_conversation
    service.save_rollup(identity, 1, 4, "确认周三王老师六年级数学" * 100)
    manager = ContextManager(identity=identity, conversation_service=service,
                             token_counter=len, summary_max_chars=150)
    result = manager.build(task="后续安排", max_tokens=100)
    assert len(result) <= 100
    assert "用户问题：后续安排" in result
    assert "近期滚动摘要" in result


def test_default_rollup_persister_stores_text_without_embedding(monkeypatch, tmp_path):
    from services.long_term_memory_service import LongTermMemoryService
    from services import text_embedding
    calls = []
    monkeypatch.setattr(text_embedding, "embed_input", lambda text: pytest.fail("rollup must not embed"))
    original = LongTermMemoryService.add_memory

    def add(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(LongTermMemoryService, "add_memory", add)
    identity = Identity(user_id="anon_context", student_id="student-7", session_id="budget-session")
    service = LongTermMemoryService(db_path=f"sqlite:///{tmp_path / 'explicit-memory.db'}")
    try:
        manager = ContextManager([history_context(rounds=6, filler="触发摘要" * 10)], identity=identity,
                                 rollup_enabled=True, rollup_rounds=2, rollup_window_ratio=0.01,
                                 memory_service=service)
        manager.build(max_tokens=10000)
        assert manager.last_rollup_triggered
        assert calls and calls[0]["auto_embed"] is False
        assert calls[0]["user_id"] == identity.user_id
        assert calls[0]["student_id"] == identity.student_id
        assert service.get_user_memories(identity.user_id, "summary", student_id=identity.student_id)
    finally:
        service.db_router.close()
