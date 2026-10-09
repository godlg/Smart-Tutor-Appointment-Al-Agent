"""入口生命周期补测：真实临时存储，工作 Agent/LLM 在边界替身。"""

import asyncio
from types import SimpleNamespace

import pytest

from config.identity import Identity, get_current_identity
from services.conversation_service import ConversationService


async def collect(stream):
    return "".join([part async for part in stream])


@pytest.fixture
def entry(tmp_path, monkeypatch):
    from api import chat_handler

    service = ConversationService(f"sqlite:///{tmp_path / 'entry.db'}")
    monkeypatch.setattr(chat_handler, "_conversation_service", service)
    for name in ("_session_agents", "_session_chat_histories", "_session_async_locks",
                 "_active_sessions", "_session_last_access"):
        monkeypatch.setattr(chat_handler, name, {}, raising=False)
    yield chat_handler, service
    service.close()


def test_explicit_identity_without_sid_uses_own_history_and_contextvar(entry, monkeypatch):
    mod, service = entry
    identity = Identity("owner", "child-a", "identity-only")
    seen = []

    class Agent:
        async def classify_task_stream(self, text):
            seen.append(get_current_identity())
            seen.append(mod.get_session_chat_history(identity.session_id))
            yield "[REPLY]回答"

    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Agent())
    asyncio.run(collect(mod.ProcessUserInput_stream("问题", identity=identity)))
    assert seen[0] == identity
    assert seen[1] is service.get_history(identity.session_id)
    assert [m.content for m in seen[1].messages] == ["问题", "[REPLY]回答"]
    assert get_current_identity() is None
    assert not mod._active_sessions


def test_capacity_evicts_only_idle_sessions_and_restores_persistence(entry, monkeypatch):
    mod, service = entry
    monkeypatch.setattr(mod, "SESSION_CACHE_MAX_ENTRIES", 2, raising=False)
    for index in range(3):
        sid = f"cache-{index}"
        identity = Identity("owner", session_id=sid)
        service.record_human(identity, str(index))
        mod.get_session_chat_history(sid)
        mod._session_agents[sid] = object()
        mod._get_session_lock(sid)
        mod._session_last_access[sid] = index + 1
    mod._enter_session("cache-0")
    mod.evict_idle_sessions(now=10)
    assert "cache-0" in mod._session_agents
    assert "cache-1" not in mod._session_agents
    assert "cache-1" not in mod._session_chat_histories
    assert "cache-1" not in mod._session_async_locks
    assert [m.content for m in service.restore_history(Identity("owner", session_id="cache-1")).messages] == ["1"]
    mod._exit_session("cache-0")


def test_waiting_cancellation_has_no_ghost_turn_and_releases_active(entry, monkeypatch):
    mod, service = entry
    identity = Identity("owner", session_id="queued")
    first_started = asyncio.Event()
    release = asyncio.Event()

    class Agent:
        async def classify_task_stream(self, text):
            if text == "q1":
                first_started.set()
                await release.wait()
            yield "[REPLY]" + text

    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Agent())

    async def scenario():
        first = asyncio.create_task(collect(mod.ProcessUserInput_stream("q1", identity=identity)))
        await first_started.wait()
        queued = asyncio.create_task(collect(mod.ProcessUserInput_stream("q2", identity=identity)))
        # 只让调度器运行一次，使 q2 等锁；不依赖墙钟延时。
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await first
        await collect(mod.ProcessUserInput_stream("q3", identity=identity))

    asyncio.run(scenario())
    rows = service.conversation_repo.get_messages(identity.session_id)
    assert [r["content"] for r in rows] == ["q1", "[REPLY]q1", "q3", "[REPLY]q3"]
    assert not mod._active_sessions


def test_stream_identity_does_not_leak_between_yields_and_cross_task_close(entry, monkeypatch):
    from config.identity import set_current_identity, reset_current_identity
    mod, service = entry
    requested = Identity("owner", session_id="cross-close")
    caller = Identity("caller", session_id="other-call")
    seen = []

    class Agent:
        async def classify_task_stream(self, text):
            seen.append(get_current_identity())
            yield "[REPLY]可用部分"
            await asyncio.Future()

    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Agent())

    async def scenario():
        caller_token = set_current_identity(caller)
        try:
            stream = mod.ProcessUserInput_stream("问题", identity=requested)
            assert await anext(stream) == "[REPLY]可用部分"
            assert get_current_identity() == caller
            await asyncio.create_task(stream.aclose())
            assert get_current_identity() == caller
        finally:
            reset_current_identity(caller_token)

    asyncio.run(scenario())
    assert seen == [requested]
    rows = service.conversation_repo.get_messages(requested.session_id)
    assert [r["role"] for r in rows] == ["human", "ai"]
    assert rows[-1]["status"] == "partial"
    assert not mod._active_sessions


def test_checkpoint_read_failure_keeps_old_draft_and_retries(entry, monkeypatch):
    from services.session_checkpoint import capture_checkpoint
    mod, service = entry
    identity = Identity("default_user", session_id="read-failure", legacy=True)
    state = SimpleNamespace(value="trial_confirm")
    source = SimpleNamespace(state=state, trial_agent=SimpleNamespace(trial_history={
        "subject": "数学", "pending_trial": {"teacher_id": 7}}))
    payload = capture_checkpoint(source)
    service.save_checkpoint(identity, "trial_confirm", payload, [])
    before = service.load_checkpoint(identity)
    agent = SimpleNamespace(state=SimpleNamespace(value="classify"))
    agent.classify_task_stream = lambda task: pytest.fail("unrestored checkpoint must not run business")
    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: agent)
    original_load = service.load_checkpoint
    attempts = []

    def flaky_load(ident):
        attempts.append(ident)
        if len(attempts) == 1:
            raise RuntimeError("transient read failure")
        return original_load(ident)

    monkeypatch.setattr(service, "load_checkpoint", flaky_load)
    with pytest.raises(RuntimeError, match="transient"):
        asyncio.run(collect(mod.ProcessUserInput_stream("确认", identity=identity)))
    assert original_load(identity) == before
    assert not getattr(agent, "_checkpoint_loaded", False)
    retry = asyncio.run(collect(mod.ProcessUserInput_stream("确认", identity=identity)))
    assert "重新确认" in retry
    assert len(attempts) == 2


@pytest.mark.parametrize("text", [
    "你好", "确认吧", "确认一下", "好的，谢谢",
    "请确认一下之前说好的明天3点数学试讲",
    "请重新确认之前说好的明天3点数学试讲",
    "不要重新预约，只确认之前说好的明天3点数学试讲",
    "你刚才让我重新预约，我只确认之前说好的明天3点数学试讲",
])
def test_irrelevant_or_confirmation_variants_cannot_disarm_restored_gate(entry, monkeypatch, text):
    mod, service = entry
    identity = Identity("default_user", session_id="gate", legacy=True)
    agent = SimpleNamespace(_checkpoint_loaded=True, _requires_reconfirmation=True,
                            _restored_drafts={"trial": {"subject": "数学"}, "lesson": {}, "recommendation": {}},
                            state=SimpleNamespace(value="classify"))
    calls = []
    async def stream(task):
        calls.append(task)
        yield "[REPLY]不应调用"
    agent.classify_task_stream = stream
    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: agent)
    output = asyncio.run(collect(mod.ProcessUserInput_stream(text, identity=identity)))
    assert "重新确认" in output
    assert calls == []
    assert agent._requires_reconfirmation is True


@pytest.mark.parametrize("fresh_request", [
    "重新预约初中数学试讲，明天下午3点",
    "好的，重新预约初中数学试讲，明天下午3点",
    "确认一下，我要重新预约初中数学试讲，明天下午3点",
])
def test_fresh_requirements_release_restored_gate_without_replaying_old_command(entry, monkeypatch, fresh_request):
    mod, service = entry
    identity = Identity("default_user", session_id="fresh-gate", legacy=True)
    agent = SimpleNamespace(
        _checkpoint_loaded=True, _requires_reconfirmation=True,
        _restored_drafts={"trial": {"subject": "数学"}, "lesson": {}, "recommendation": {}},
        state=SimpleNamespace(value="classify"),
    )
    calls = []

    async def stream(task):
        calls.append(task)
        yield "[REPLY]重新查询后请确认"

    agent.classify_task_stream = stream
    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: agent)
    output = asyncio.run(collect(mod.ProcessUserInput_stream(fresh_request, identity=identity)))
    assert calls == [fresh_request]
    assert output == "[REPLY]重新查询后请确认"
    assert agent._requires_reconfirmation is False
    assert service.load_checkpoint(identity) is not None


def test_stream_identity_is_scoped_on_resume_and_business_cleanup(entry, monkeypatch):
    from config.identity import set_current_identity, reset_current_identity
    mod, service = entry
    requested = Identity("owner", session_id="resume-cleanup")
    caller = Identity("caller", session_id="caller-resume")
    seen = []

    class Agent:
        async def classify_task_stream(self, text):
            try:
                seen.append(get_current_identity())
                yield "[REPLY]第一段"
                seen.append(get_current_identity())
                yield "第二段"
            finally:
                seen.append(get_current_identity())

    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Agent())

    async def scenario():
        caller_token = set_current_identity(caller)
        try:
            stream = mod.ProcessUserInput_stream("问题", identity=requested)
            assert await anext(stream) == "[REPLY]第一段"
            assert get_current_identity() == caller
            assert await asyncio.create_task(anext(stream)) == "第二段"
            assert get_current_identity() == caller
            await asyncio.create_task(stream.aclose())
            assert get_current_identity() == caller
        finally:
            reset_current_identity(caller_token)

    asyncio.run(scenario())
    assert seen == [requested, requested, requested]
    rows = service.conversation_repo.get_messages(requested.session_id)
    assert rows[-1]["content"] == "[REPLY]第一段第二段"
    assert rows[-1]["status"] == "partial"
    assert not mod._active_sessions
    assert not mod._get_session_lock(requested.session_id).locked()


def test_steady_writes_only_load_unsynchronized_message_suffix(entry, monkeypatch):
    _, service = entry
    identity = Identity("owner", session_id="incremental")
    original = service.conversation_repo.get_messages
    offsets = []

    def observe(session_id, after_seq=0):
        offsets.append(after_seq)
        if after_seq:
            return original(session_id, after_seq=after_seq)
        return original(session_id)

    monkeypatch.setattr(service.conversation_repo, "get_messages", observe)
    service.record_human(identity, "第一句")
    service.record_ai(identity, "回答")
    service.record_human(identity, "第三句")
    assert offsets == [0, 1, 2]
    assert [m.content for m in service.get_history(identity.session_id).messages] == ["第一句", "回答", "第三句"]


def test_entry_saves_checkpoint_and_restore_requires_fresh_confirmation(entry, monkeypatch):
    mod, service = entry
    identity = Identity("default_user", session_id="pending-entry", legacy=True)

    def make_agent():
        state = SimpleNamespace(value="classify")
        child = SimpleNamespace(trial_history={}, finished=False)
        agent = SimpleNamespace(state=state, state_manager=SimpleNamespace(state=state),
                                trial_agent=child, session_id=identity.session_id)
        agent.calls = []

        async def stream(text):
            agent.calls.append(text)
            child.trial_history = {"subject": "数学", "grade_level": "初中",
                                   "start_time": "2030-10-03 15:00",
                                   "pending_trial": {"teacher_id": 7}}
            state.value = "trial_confirm"
            yield "[REPLY]请确认试讲草稿"
        agent.classify_task_stream = stream
        return agent

    source = make_agent()
    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: source)
    asyncio.run(collect(mod.ProcessUserInput_stream("数学初中，明天下午", identity=identity)))
    checkpoint = service.load_checkpoint(identity)
    assert checkpoint is not None
    assert checkpoint["state"] == "trial_confirm"
    mod.drop_session(identity.session_id)
    restored = make_agent()
    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: restored)
    output = asyncio.run(collect(mod.ProcessUserInput_stream("确认", identity=identity)))
    assert "重新确认" in output
    assert restored.calls == []  # 旧确认没有被交给任何业务执行者。
    assert restored.trial_agent.trial_history["subject"] == "数学"
    assert restored.trial_agent.trial_history.get("pending_trial") is None
    assert restored.trial_agent.trial_history["start_time"] is None
    assert len(service.restore_history(identity).messages) == 4
