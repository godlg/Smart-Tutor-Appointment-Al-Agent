"""Incremental context/consultation reliability checks, always offline and isolated."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.context.contexts.conversation import ConversationContext
from agents.context.contexts.memory import MemoryContext
from agents.context.manager import ContextManager
from agents.context.types import ContextBudgetExceeded, estimate_token_count
from config.identity import Identity
from services.conversation_service import ConversationService


@pytest.fixture
def conversation(tmp_path):
    service = ConversationService(db_path=f"sqlite:///{tmp_path / 'context-extra.db'}")
    yield service
    service.close()


def add_rounds(service, identity, rounds=10):
    for index in range(rounds):
        service.record_human(identity, f"旧问题{index:02d}")
        service.record_ai(identity, f"旧回答{index:02d}")


def test_atomic_rollup_does_not_construct_second_default_memory_service(conversation, monkeypatch):
    from services.long_term_memory_service import LongTermMemoryService
    identity = Identity("anon-extra", "child", "rollup-extra")
    add_rounds(conversation, identity)
    opened = []
    original = LongTermMemoryService.__init__

    def track(self, *args, **kwargs):
        opened.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(LongTermMemoryService, "__init__", track)
    conv = ConversationContext(history_provider=conversation.get_history)
    manager = ContextManager([conv], identity=identity, conversation_service=conversation,
                             rollup_enabled=True, rollup_window_ratio=0.01, rollup_rounds=2)
    result = manager.build(max_tokens=10000)
    assert manager.last_rollup_triggered
    assert conversation.summary_watermark(identity) == 4
    assert "近期滚动摘要" in result
    summaries = conversation.memory_service.get_user_memories(
        identity.user_id, "summary", student_id=identity.student_id)
    assert len(summaries) == 1
    assert opened == [], "archive owner must not create another default-DB memory copy"


def test_profile_lazy_analyzer_uses_memory_services_shared_router(conversation, monkeypatch):
    identity = Identity("anon-extra", "child", "profile-extra")
    repo = conversation.db_router.user_behavior_repo
    repo.record_behavior(identity.memory_scope(), "lesson", {"subject": "物理", "duration_minutes": 45})
    repo.record_behavior("default_user", "lesson", {"subject": "英语"})
    repo.record_behavior(Identity(identity.user_id, "other").memory_scope(), "lesson", {"subject": "语文"})
    monkeypatch.setattr("services.user_behavior_service.UserBehaviorService.__init__",
                        lambda *args, **kwargs: pytest.fail("must reuse injected router"))
    context = MemoryContext(identity=identity, memory_service=conversation.memory_service)
    text = context.build_profile_summary()
    assert "偏好学科: 物理" in text
    assert "英语" not in text and "语文" not in text
    assert context.analyzer.behavior_service is repo


def test_identity_only_memory_context_never_opens_default_services(monkeypatch):
    opened = []
    monkeypatch.setattr("services.user_behavior_service.UserBehaviorService.__init__",
                        lambda *args, **kwargs: opened.append("behavior"))
    monkeypatch.setattr("services.long_term_memory_service.LongTermMemoryService.__init__",
                        lambda *args, **kwargs: opened.append("memory"))
    context = MemoryContext(identity=Identity("anon-extra", "child", "identity-only"))
    text = context.build(SimpleNamespace(task="学习建议"))
    assert "暂无用户画像" in text
    assert opened == [], "explicit identity must fail closed without its storage dependency"


def test_no_vector_summary_has_independent_quota_with_five_vector_hits(conversation, monkeypatch):
    identity = Identity("anon-extra", "child", "quota-extra")
    service = conversation.memory_service
    for index in range(5):
        service.add_memory(identity.user_id, "preference", f"向量记忆{index}",
                           embedding=[1.0, 0.0], student_id=identity.student_id)
    service.add_memory(identity.user_id, "summary", "确认周三19:00王老师只能线上",
                       auto_embed=False, student_id=identity.student_id)
    service.add_memory(identity.user_id, "summary", "另一学员的私有摘要",
                       auto_embed=False, student_id="other")
    calls = []
    from services import text_embedding
    monkeypatch.setattr(text_embedding, "embed_input", lambda query: calls.append(query) or [1.0, 0.0])
    context = MemoryContext(identity=identity, memory_service=service, analyzer=MagicMock())
    context.analyzer.analyze_user_preferences.return_value = None
    text = ContextManager([context]).build(task="学习安排")
    assert all(f"向量记忆{i}" in text for i in range(5))
    assert "确认周三19:00王老师只能线上" in text
    assert "另一学员的私有摘要" not in text
    assert calls == ["学习安排"]


def test_summary_text_survives_vector_retrieval_exception(conversation, monkeypatch):
    identity = Identity("anon-extra", None, "vector-failure")
    service = conversation.memory_service
    service.add_memory(identity.user_id, "summary", "确认周四数学课", auto_embed=False)
    monkeypatch.setattr(service, "retrieve_top_k", MagicMock(side_effect=RuntimeError("vector down")))
    analyzer = MagicMock()
    analyzer.analyze_user_preferences.return_value = None
    text = MemoryContext(identity=identity, memory_service=service, analyzer=analyzer).build(
        SimpleNamespace(task="安排"))
    assert "确认周四数学课" in text


def test_text_summary_does_not_require_query_or_embedding(conversation, monkeypatch):
    identity = Identity("anon-extra", "child", "no-query")
    service = conversation.memory_service
    service.add_memory(identity.user_id, "summary", "确认周三19:00王老师", auto_embed=False,
                       student_id=identity.student_id)
    monkeypatch.setattr(service, "retrieve_top_k", lambda *args, **kwargs: pytest.fail("no query embedding"))
    analyzer = MagicMock()
    analyzer.analyze_user_preferences.return_value = None
    text = MemoryContext(identity=identity, memory_service=service, analyzer=analyzer).build(
        SimpleNamespace(task=""))
    assert "确认周三19:00王老师" in text
    assert "近期记忆摘要" in text


def test_extractive_summary_shares_space_across_constraints_not_one_long_sentence():
    manager = ContextManager(summary_max_chars=180)
    messages = [HumanMessage(content="确认只能线上，不能周末，必须先沟通，预约时间待办：" + "闲聊" * 1000),
                HumanMessage(content="六年级数学"), HumanMessage(content="王老师"),
                HumanMessage(content="周三19:00"), AIMessage(content="待办预约试讲")]
    summary = manager._summarize(messages)
    assert len(summary) <= 180
    assert "extractive" in summary
    for fact in ("只能线上", "不能周末", "六年级数学", "王老师", "周三19:00", "待办"):
        assert fact in summary


def test_default_cjk_estimator_is_conservative_for_multitoken_characters():
    assert estimate_token_count("龘" * 20) >= 40
    assert estimate_token_count("你好数学") >= 4
    assert estimate_token_count("") == 0


def make_consultant(monkeypatch, **kwargs):
    from agents.consultant_agent import ConsultantAgent
    monkeypatch.setattr(ConsultantAgent, "_initialize_llm", lambda self: MagicMock())
    monkeypatch.setattr("agents.consultant_agent.KnowledgeRetriever", lambda: MagicMock())
    return ConsultantAgent(**kwargs)


def test_identity_consultant_without_service_has_no_legacy_default_profile(monkeypatch):
    identity = Identity("anon-extra", "child", "isolated-consultant")
    agent = make_consultant(monkeypatch, identity=identity)
    assert agent.session_id == identity.session_id
    memory = next(ctx for ctx in agent.context_manager.contexts if isinstance(ctx, MemoryContext))
    assert memory.identity == identity
    assert agent.consultation_processor.profile_provider() == (None, None)


def test_rag_history_is_bounded_and_includes_recent_durable_summary(conversation, monkeypatch):
    identity = Identity("anon-extra", "child", "rag-extra")
    for index in range(30):
        conversation.record_human(identity, f"远古问题{index:02d}" + "闲聊" * 400)
        conversation.record_ai(identity, f"远古回答{index:02d}" + "解释" * 400)
    conversation.save_rollup(identity, 1, 4, "确认六年级数学王老师周三19:00只能线上")
    conversation.record_human(identity, "近期问题")
    conversation.record_ai(identity, "近期回答")
    agent = make_consultant(monkeypatch, identity=identity,
                           session_chat_history=conversation.get_history(identity.session_id),
                           conversation_service=conversation)
    text = agent._build_rag_query_context()
    assert len(text) <= 1600
    assert "近期滚动摘要" in text
    assert "确认六年级数学王老师周三19:00只能线上" in text
    assert "近期问题" in text and "近期回答" in text
    assert "远古问题00" not in text and "远古问题20" not in text


def test_consultation_behavior_without_injected_storage_does_not_fallback_default_db(monkeypatch, caplog):
    from agents.consultant.consultation_processor import ConsultationProcessor
    identity = Identity("anon-extra", "child", "behavior-missing")
    opened = []
    monkeypatch.setattr("services.user_behavior_service.UserBehaviorService.__init__",
                        lambda *args, **kwargs: opened.append(True))
    processor = ConsultationProcessor(MagicMock(), MagicMock(), MagicMock(), identity=identity)
    with caplog.at_level(logging.WARNING):
        asyncio.run(processor._record_consultation_behavior("问题", [], identity.session_id))
    assert opened == []
    assert "行为" in caplog.text


def test_consultation_behavior_writes_identity_scope_to_shared_router(conversation):
    from agents.consultant.consultation_processor import ConsultationProcessor
    identity = Identity("anon-extra", "child", "behavior-extra")
    repo = conversation.db_router.user_behavior_repo
    processor = ConsultationProcessor(MagicMock(), MagicMock(), MagicMock(),
                                      identity=identity, behavior_service=repo)
    asyncio.run(processor._record_consultation_behavior("数学怎么学", [{"category": "学习"}], "ignored"))
    records = repo.get_user_behaviors(identity.memory_scope(), "consultation")
    assert len(records) == 1
    assert records[0]["session_id"] == identity.session_id
    assert records[0]["action_data"]["question"] == "数学怎么学"
    assert repo.get_user_behaviors("default_user") == []
    assert repo.get_user_behaviors(identity.user_id) == []


def test_processor_handles_budget_exception_as_clear_retry_not_unrelated_or_internal_error():
    from agents.consultant.consultation_processor import ConsultationProcessor
    retriever = MagicMock()
    retriever.search_knowledge = AsyncMock(return_value=[])
    generator = MagicMock()

    async def budget_error(*args):
        raise ContextBudgetExceeded("internal budget configuration detail", 100, 10)
        yield  # async generator boundary

    generator.generate_response_stream = budget_error
    processor = ConsultationProcessor(retriever, MagicMock(), generator)

    async def collect():
        return "".join([token async for token in processor.process_consultation_stream("问题", "budget")])

    text = asyncio.run(collect())
    assert "请缩短输入" in text
    assert "internal budget" not in text
    assert "无关" not in text and "转回" not in text


def test_identity_rollup_without_archive_dependency_never_opens_default_memory(monkeypatch, caplog):
    from services.long_term_memory_service import LongTermMemoryService
    opened = []
    monkeypatch.setattr(LongTermMemoryService, "__init__", lambda *args, **kwargs: opened.append(True))
    conv = ConversationContext()
    identity = Identity("anon-extra", "child", "rollup-no-storage")
    for index in range(6):
        conv.get_history(identity.session_id).add_messages([
            HumanMessage(content=f"原文{index}"), AIMessage(content="回复")])
    manager = ContextManager([conv], identity=identity, rollup_enabled=True,
                             rollup_rounds=2, rollup_window_ratio=0.01)
    with caplog.at_level(logging.WARNING):
        text = manager.build(max_tokens=10000)
    assert "原文0" in text
    assert not manager.last_rollup_triggered
    assert conv._rolled_up_count(identity.session_id) == 0
    assert opened == []
    assert "水位不推进" in caplog.text


def test_foreign_identity_context_cannot_read_cached_owned_history(conversation):
    from services.identity_service import SessionOwnershipError
    owner = Identity("owner-extra", None, "owned-context")
    conversation.record_human(owner, "owner-private-message")
    attacker = Identity("other-extra", None, owner.session_id)
    manager = ContextManager([ConversationContext(history_provider=conversation.get_history)],
                             identity=attacker, conversation_service=conversation)
    with pytest.raises(SessionOwnershipError):
        manager.build(task="读取")
    with pytest.raises(SessionOwnershipError):
        manager.build_retrieval_context()


def test_unknown_ownership_due_to_storage_failure_cannot_read_cached_history(conversation, monkeypatch):
    owner = Identity("owner-extra", None, "storage-unknown")
    conversation.record_human(owner, "cached-private-message")
    attacker = Identity("other-extra", None, owner.session_id)
    monkeypatch.setattr(conversation, "ensure_session", MagicMock(side_effect=RuntimeError("storage down")))
    provider = MagicMock(wraps=conversation.get_history)
    manager = ContextManager([ConversationContext(history_provider=provider)],
                             identity=attacker, conversation_service=conversation)
    with pytest.raises(RuntimeError, match="storage down"):
        manager.build(task="读取")
    with pytest.raises(RuntimeError, match="storage down"):
        manager.build_retrieval_context()
    provider.assert_not_called()


def test_identity_consultant_without_service_refuses_unbound_history(monkeypatch):
    from langchain_core.chat_history import InMemoryChatMessageHistory
    private_history = InMemoryChatMessageHistory(messages=[HumanMessage(content="foreign-private")])
    with pytest.raises(ValueError, match="历史|存储"):
        make_consultant(monkeypatch, identity=Identity("anon-extra", "child", "unbound"),
                        session_chat_history=private_history)


@pytest.mark.parametrize("context_identity", [None, Identity("other-extra", "child"),
                                               Identity("anon-extra", "other-child")])
def test_memory_component_scope_must_match_identity_manager(context_identity, monkeypatch):
    identity = Identity("anon-extra", "child", "bound-manager")
    reads = []
    service = MagicMock()
    service.retrieve_top_k.side_effect = lambda **kwargs: reads.append(kwargs) or []
    service.get_user_memories.side_effect = lambda *args, **kwargs: reads.append(kwargs) or []
    analyzer = MagicMock()
    context = MemoryContext(identity=context_identity, memory_service=service, analyzer=analyzer)
    manager = ContextManager([context], identity=identity)
    with pytest.raises(ValueError, match="scope|身份"):
        manager.build(task="读取")
    assert reads == []
    analyzer.analyze_user_preferences.assert_not_called()


@pytest.mark.parametrize("method", ["build", "build_retrieval_context"])
def test_identity_manager_without_service_refuses_external_history_provider(method):
    from langchain_core.chat_history import InMemoryChatMessageHistory
    provider = MagicMock(return_value=InMemoryChatMessageHistory(
        messages=[HumanMessage(content="foreign-private")]))
    manager = ContextManager([ConversationContext(history_provider=provider)],
                             identity=Identity("anon-extra", "child", "unbound-manager"))
    with pytest.raises(ValueError, match="历史|存储"):
        getattr(manager, method)()
    provider.assert_not_called()


def test_consultant_profile_helper_validates_scope_before_any_profile_read(monkeypatch):
    identity = Identity("anon-extra", "child", "profile-helper")
    agent = make_consultant(monkeypatch, identity=identity)
    analyzer = MagicMock()
    foreign = MemoryContext(identity=Identity("other-extra", "child"), analyzer=analyzer,
                            memory_service=MagicMock())
    agent.context_manager.contexts = [foreign]
    with pytest.raises(ValueError, match="scope|身份"):
        agent._build_memory_profile_text()
    with pytest.raises(ValueError, match="scope|身份"):
        agent._build_rag_query_context()
    analyzer.analyze_user_preferences.assert_not_called()
    foreign._memory_service.retrieve_top_k.assert_not_called()


def test_processor_hides_internal_retrieval_exception_and_does_not_record_success():
    from agents.consultant.consultation_processor import ConsultationProcessor
    retriever = MagicMock()
    retriever.search_knowledge = AsyncMock(side_effect=RuntimeError("internal-path-secret-sentinel"))
    generator = MagicMock()
    processor = ConsultationProcessor(retriever, MagicMock(), generator)
    processor._record_consultation_behavior = AsyncMock()

    async def collect():
        return "".join([token async for token in processor.process_consultation_stream("问题", "error")])

    text = asyncio.run(collect())
    assert "请稍后重试" in text
    assert "internal-path-secret-sentinel" not in text
    generator.generate_response_stream.assert_not_called()
    processor._record_consultation_behavior.assert_not_awaited()


def test_tight_summary_budget_preserves_newest_correction_before_old_constraint(conversation):
    identity = Identity("anon-extra", None, "summary-correction")
    add_rounds(conversation, identity, rounds=3)
    conversation.save_rollup(identity, 1, 2, "旧约束周三19:00" * 8)
    conversation.save_rollup(identity, 3, 4, "改为周四19:00，不能周三")
    manager = ContextManager(identity=identity, conversation_service=conversation,
                             token_counter=len, summary_max_chars=300)
    text = manager.build(task="安排", max_tokens=70)
    assert len(text) <= 70
    assert "改为周四19:00，不能周三" in text
    assert "改为周四19:00，不能周三" in manager.build_retrieval_context(max_chars=100)


def test_summary_quota_is_filled_after_deduplicating_vector_summaries(conversation, monkeypatch):
    identity = Identity("anon-extra", "child", "overlapping-summary")
    service = conversation.memory_service
    service.add_memory(identity.user_id, "summary", "无向量确认约束", auto_embed=False,
                       student_id=identity.student_id)
    for index in range(3):
        service.add_memory(identity.user_id, "summary", f"向量旧摘要{index}", embedding=[1.0, 0.0],
                           student_id=identity.student_id)
    from services import text_embedding
    monkeypatch.setattr(text_embedding, "embed_input", lambda query: [1.0, 0.0])
    analyzer = MagicMock()
    analyzer.analyze_user_preferences.return_value = None
    text = MemoryContext(identity=identity, memory_service=service, analyzer=analyzer).build(
        SimpleNamespace(task="安排"))
    assert "无向量确认约束" in text
    assert "近期记忆摘要" in text
    assert all(text.count(f"向量旧摘要{index}") == 1 for index in range(3))


def test_real_stream_generator_budget_rejection_skips_consultation_behavior():
    from agents.consultant.response_generator import ResponseGenerator
    from agents.consultant.consultation_processor import ConsultationProcessor
    model = MagicMock()
    model.ainvoke = AsyncMock()
    manager = ContextManager(token_counter=len)
    manager.max_tokens = 5
    generator = ResponseGenerator(model, context_manager=manager)
    retriever = MagicMock()
    retriever.search_knowledge = AsyncMock(return_value=[])
    processor = ConsultationProcessor(retriever, MagicMock(), generator)
    processor._record_consultation_behavior = AsyncMock()

    async def collect():
        return "".join([token async for token in processor.process_consultation_stream("问题", "budget")])

    text = asyncio.run(collect())
    assert "请缩短输入" in text
    model.ainvoke.assert_not_awaited()
    model.astream.assert_not_called()
    processor._record_consultation_behavior.assert_not_awaited()


def test_response_budget_rejection_never_calls_model():
    from agents.consultant.response_generator import ResponseGenerator
    model = MagicMock()
    model.ainvoke = AsyncMock()
    manager = ContextManager(token_counter=len)
    manager.max_tokens = 5
    generator = ResponseGenerator(model, context_manager=manager)
    text = asyncio.run(generator.generate_response("必须保留的大问题", []))
    assert "请缩短输入" in text
    model.ainvoke.assert_not_awaited()
