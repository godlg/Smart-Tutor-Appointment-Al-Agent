"""真实咨询→router→主入口调用链的屏障、持久化、取消和身份验证。"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from test_consultation_real_stream import BarrierModel, FIRST, LAST, PREFIX, collect


class ConsultationModel(BarrierModel):
    async def ainvoke(self, messages, **kwargs):
        self.invoke_calls.append(messages)
        assert "YES" in messages[0]["content"], "answer generation fell back to ainvoke"
        return AIMessage(content="YES")


@pytest.fixture
def chain_env(tmp_path, monkeypatch):
    from api import chat_handler
    from agents import consultant_agent, task_classification_agent
    from services.conversation_service import ConversationService
    from services import text_embedding
    from config.identity import Identity
    from db.base.session_manager import SessionManager

    managers = []
    original = SessionManager.__init__

    def track(manager, *args, **kwargs):
        original(manager, *args, **kwargs)
        managers.append(manager)

    monkeypatch.setattr(SessionManager, "__init__", track)
    monkeypatch.setattr(text_embedding, "create_embedding_model", lambda: type(
        "Embedding", (), {"embed_query": lambda self, text: [1.0, 0.0, 0.0]})())
    monkeypatch.setattr(task_classification_agent, "create_chat_model",
                        lambda **kwargs: FakeListChatModel(responses=["consultation"]))
    service = ConversationService(db_path=f"sqlite:///{tmp_path / 'stream.db'}")
    monkeypatch.setattr(chat_handler, "_conversation_service", service)

    def build(model, sid="real-stream", user="owner-a", student="child-a", continued=False):
        identity = service.ensure_session(Identity(user, student_id=student, session_id=sid))
        history = service.restore_history(identity)
        monkeypatch.setattr(consultant_agent, "create_chat_model", lambda **kwargs: model)
        consultant = consultant_agent.ConsultantAgent(
            session_id=sid, identity=identity, conversation_service=service,
            session_chat_history=history)
        agent = task_classification_agent.TaskClassificationAgent(
            consultant_agent=consultant, session_id=sid,
            session_chat_history=history, context_manager=consultant.context_manager)
        agent.identity = identity
        if continued:
            agent.state_manager.transition_to_consultation()
        chat_handler._session_agents[sid] = agent
        return identity, agent

    yield chat_handler, service, build
    for manager in reversed(managers):
        manager.close()


async def through_first_body(stream):
    parts = []
    import time as _t
    _s = _t.monotonic()
    while FIRST not in "".join(parts):
        _t0 = _t.monotonic()
        parts.append(await asyncio.wait_for(anext(stream), 3))
        print(f"### anext +{_t.monotonic()-_t0:.2f}s (total {_t.monotonic()-_s:.2f}s): {parts[-1]!r}", flush=True)
    return "".join(parts)


def behaviors(service, identity):
    return service.db_router.user_behavior_repo.get_user_behaviors(
        identity.memory_scope(), action_type="consultation")


def assert_saved(service, identity, status, body):
    rows = service.conversation_repo.get_messages(identity.session_id)
    assert [row["role"] for row in rows] == ["human", "ai"]
    assert rows[1]["status"] == status and body in rows[1]["content"]
    assert "[THOUGHT]" not in rows[1]["content"]
    history = service.restore_history(identity)
    assert len(history.messages) == 2
    assert len(service.restore_history(identity).messages) == 2
    return rows


@pytest.mark.parametrize("continued", [False, True])
def test_entry_receives_body_before_completion_and_saves_once(chain_env, continued):
    mod, service, build = chain_env

    async def scenario():
        model = ConsultationModel(blocked=True)
        identity, agent = build(model, continued=continued)
        stream = mod.ProcessUserInput_stream("初二数学怎么学", identity=identity)
        try:
            first = await through_first_body(stream)
            assert PREFIX + FIRST in first
            assert not model.done and not model.release.is_set()
            assert len(model.invoke_calls) == 1  # 仍只有咨询 YES/NO 调用 ainvoke
            assert behaviors(service, identity) == []
            tail = asyncio.create_task(collect(stream))
            await asyncio.wait_for(model.waiting.wait(), 3)
            assert not tail.done()
            model.release.set()
            assert await tail == LAST
            assert_saved(service, identity, "complete", FIRST + LAST)
            assert len(behaviors(service, identity)) == 1
            assert not mod._get_session_lock(identity.session_id).locked()
            assert identity.session_id not in mod._active_sessions
            assert agent.state_manager.get_current_state() == "classify"
        finally:
            model.release.set()
            await stream.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("fail,chunks", [(0, None), (1, None), (2, None), (None, [])])
def test_entry_failure_is_partial_and_not_success(chain_env, fail, chunks):
    mod, service, build = chain_env
    model = ConsultationModel(chunks, fail=fail)
    identity, _ = build(model)
    result = asyncio.run(collect(mod.ProcessUserInput_stream("初二数学怎么学", identity=identity)))
    assert result.count(PREFIX) == 1 and result.count("[ERROR]") == 1
    assert "SECRET_RAW_MODEL_ERROR" not in result
    rows = assert_saved(service, identity, "partial", "[ERROR]")
    if fail in (1, 2):
        assert FIRST in rows[1]["content"]
    assert behaviors(service, identity) == []
    assert not mod._get_session_lock(identity.session_id).locked()
    assert identity.session_id not in mod._active_sessions


@pytest.mark.parametrize("continued", [False, True])
@pytest.mark.parametrize("mode", ["close", "cancel"])
def test_close_or_cancel_cleans_underlying_model_before_loop_shutdown(chain_env, mode, continued):
    mod, service, build = chain_env
    from config.identity import Identity, get_current_identity, set_current_identity, reset_current_identity

    async def scenario():
        model = ConsultationModel(blocked=True)
        identity, _ = build(model, continued=continued)
        caller = Identity("caller", session_id="caller-context")
        token = set_current_identity(caller)
        stream = mod.ProcessUserInput_stream("初二数学怎么学", identity=identity)
        try:
            await through_first_body(stream)
            assert get_current_identity() == caller
            if mode == "close":
                await asyncio.create_task(stream.aclose())  # 另一个 task 关闭，ContextVar 仍配对
            else:
                pending = asyncio.create_task(collect(stream))
                await asyncio.wait_for(model.waiting.wait(), 3)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            # 必须在 asyncio.run 的异步生成器垃圾收尾之前已清理。
            assert model.closed.is_set(), "model stream left open after entry closed"
            assert not model.done and not model.release.is_set()
            assert get_current_identity() == caller
            assert all(item == identity for item in model.identities)
            assert_saved(service, identity, "partial", FIRST)
            assert behaviors(service, identity) == []
            assert not mod._get_session_lock(identity.session_id).locked()
            assert identity.session_id not in mod._active_sessions
        finally:
            model.release.set()
            await stream.aclose()
            reset_current_identity(token)
    asyncio.run(scenario())


def test_interleaved_streams_never_leak_identity_or_history(chain_env):
    mod, service, build = chain_env
    from config.identity import get_current_identity

    async def scenario():
        models = [ConsultationModel(blocked=True), ConsultationModel(blocked=True)]
        identities = [build(models[0], sid="stream-a")[0],
                      build(models[1], sid="stream-b", user="owner-b", student="child-b")[0]]
        streams = [mod.ProcessUserInput_stream("初二数学怎么学", identity=ident) for ident in identities]
        try:
            for stream in streams:
                await through_first_body(stream)
                assert get_current_identity() is None
            for stream in reversed(streams):
                await asyncio.create_task(stream.aclose())
            for model, ident in zip(models, identities):
                assert model.closed.is_set()
                assert all(item == ident for item in model.identities)
                assert_saved(service, ident, "partial", FIRST)
            from services.identity_service import SessionOwnershipError
            with pytest.raises(SessionOwnershipError):
                # 持久 session 不能被其它 owner 接管。
                service.ensure_session(identities[1].with_session("stream-a"))
        finally:
            for model, stream in zip(models, streams):
                model.release.set()
                await stream.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_web_body_close_explicitly_closes_real_entry(chain_env, monkeypatch, path):
    mod, service, build = chain_env
    from api import identity as identity_api
    from config.identity import IDENTITY_COOKIE_NAME, get_current_identity
    from services.identity_service import IdentityService
    from starlette.requests import Request
    from web import routes

    identities = IdentityService(db_router=service.db_router)
    monkeypatch.setattr(identity_api, "_identity_service", identities)
    owner = identities.issue_new_identity()

    async def scenario():
        model = ConsultationModel(blocked=True)
        identity, _ = build(model, user=owner.user_id, student=None)
        request = Request({"type": "http", "method": "POST", "scheme": "http", "path": path,
                           "headers": [(b"cookie", f"{IDENTITY_COOKIE_NAME}={owner.user_id}".encode())]})
        response = routes._chat_response(routes.ChatRequest(
            message="初二数学怎么学", session_id=identity.session_id), request)
        stream = response.body_iterator
        try:
            await through_first_body(stream)
            await stream.aclose()
            assert model.closed.is_set(), "web wrapper did not close the entry stream"
            assert_saved(service, identity, "partial", FIRST)
            assert not mod._get_session_lock(identity.session_id).locked()
            assert get_current_identity() is None
        finally:
            model.release.set()
            await stream.aclose()
    asyncio.run(scenario())


def test_checkpoint_save_failure_does_not_break_partial_or_lock_release(chain_env, monkeypatch):
    mod, service, build = chain_env
    from config.identity import get_current_identity

    async def scenario():
        model = ConsultationModel(blocked=True)
        identity, _ = build(model)
        monkeypatch.setattr(service, "save_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("checkpoint unavailable")))
        stream = mod.ProcessUserInput_stream("初二数学怎么学", identity=identity)
        try:
            await through_first_body(stream)
            await stream.aclose()
            assert model.closed.is_set()
            assert_saved(service, identity, "partial", FIRST)
            assert not mod._get_session_lock(identity.session_id).locked()
            assert identity.session_id not in mod._active_sessions
            assert get_current_identity() is None
        finally:
            model.release.set()
            await stream.aclose()
    asyncio.run(scenario())
