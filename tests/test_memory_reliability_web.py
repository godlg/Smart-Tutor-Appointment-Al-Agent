"""真实主控/咨询/Web纵向切片。只mock LLM与embedding边界,DB均为临时库。"""
import asyncio
from contextlib import aclosing
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field


class RecordingModel(FakeListChatModel):
    prompts: list[str] = Field(default_factory=list)

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _acall(self, messages, stop=None, run_manager=None, **kwargs):
        return self._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # 真流式不经过 _call；捕获同一输入，保留原有历史/记忆隔离断言。
        self.prompts.append("\n".join(str(m.content) for m in messages))
        async with aclosing(super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )) as stream:
            async for chunk in stream:
                yield chunk


async def collect(stream):
    return "".join([part async for part in stream])


@pytest.fixture
def live_chat(tmp_path, monkeypatch):
    from api import chat_handler, identity as identity_api
    from services.conversation_service import ConversationService
    from services.identity_service import IdentityService
    from agents import consultant_agent, task_classification_agent, recommendation_agent
    from agents.trial import trial_agent
    from agents.lesson import lesson_agent
    from services import text_embedding, knowledge_service, hybrid_retriever

    url = f"sqlite:///{tmp_path / 'live.db'}"
    conversation = ConversationService(url)
    identities = IdentityService(url)
    classifier_model = RecordingModel(responses=["consultation"])
    consultant_model = RecordingModel(responses=["YES", "建议复习初中数学，每周三练习。"])
    monkeypatch.setattr(consultant_agent.ConsultantAgent, "_initialize_llm", lambda self: consultant_model)
    for module in (task_classification_agent, recommendation_agent, trial_agent, lesson_agent):
        monkeypatch.setattr(module, "create_chat_model", lambda **kw: classifier_model)
    for module in (text_embedding, knowledge_service, hybrid_retriever):
        monkeypatch.setattr(module, "embed_input", lambda text: [1.0, 0.0])
    monkeypatch.setattr(chat_handler, "_conversation_service", conversation)
    monkeypatch.setattr(identity_api, "_identity_service", identities)
    for name in ("_session_agents", "_session_chat_histories", "_session_async_locks", "_active_sessions", "_session_last_access"):
        monkeypatch.setattr(chat_handler, name, {}, raising=False)
    monkeypatch.setattr(chat_handler, "_task_agent", None)
    yield chat_handler, conversation, identities, classifier_model, consultant_model
    conversation.close()
    identities.close()


def test_thought_and_reply_same_line_are_parsed_by_role():
    from services.conversation_service import strip_thought_output
    raw = "[THOUGHT][归类机器人]路由中[REPLY][咨询机器人]你好\n第二段[THOUGHT]调试[REPLY]结尾"
    assert strip_thought_output(raw) == "[REPLY][咨询机器人]你好\n第二段[REPLY]结尾"


def test_real_consultation_history_and_identity_reach_prompt_and_behavior(live_chat):
    from config.identity import Identity
    from db.models import UserBehavior
    mod, svc, _, classifier, consultant = live_chat
    ident = Identity(user_id="anon_parent", student_id="child-a", session_id="live-consult")
    asyncio.run(collect(mod.ProcessUserInput_stream("孩子初二数学薄弱", identity=ident)))
    asyncio.run(collect(mod.ProcessUserInput_stream("每周三怎么安排练习", identity=ident)))
    rows = svc.conversation_repo.get_messages(ident.session_id)
    assert [r["role"] for r in rows] == ["human", "ai", "human", "ai"]
    assert "建议复习" in rows[1]["content"]
    assert all("[THOUGHT]" not in row["content"] for row in rows)
    history = mod.get_session_chat_history(ident.session_id)
    assert history is svc.get_history(ident.session_id)
    assert len(history.messages) == 4
    assert "孩子初二数学薄弱" in classifier.prompts[-1]
    assert "孩子初二数学薄弱" in consultant.prompts[-1]
    assert "建议复习" in consultant.prompts[-1]
    agent = mod.get_task_agent(ident.session_id)
    assert agent.consultant_agent.context_manager.rollup_enabled is True
    assert agent.consultant_agent.context_manager.max_tokens > 0
    with svc.db_router.session_manager.session_scope() as session:
        behaviors = session.query(UserBehavior).filter(UserBehavior.session_id == ident.session_id).all()
        assert len(behaviors) == 2
        assert all(b.user_id == ident.memory_scope() for b in behaviors)


def test_memory_scope_never_falls_back_to_default_student(live_chat):
    from config.identity import Identity
    from agents.context.contexts.memory import MemoryContext
    mod, svc, _, _, consultant = live_chat
    memories = svc.memory_service
    memories.add_memory("default_user", "preference", "LEGACY_SECRET", embedding=[1.0, 0.0], auto_embed=False)
    memories.add_memory("anon_parent", "preference", "CHILD_A_ONLY", embedding=[1.0, 0.0], auto_embed=False, student_id="child-a")
    memories.add_memory("anon_parent", "preference", "CHILD_B_ONLY", embedding=[1.0, 0.0], auto_embed=False, student_id="child-b")
    memories.add_memory("anon_parent", "preference", "FAMILY_ONLY", embedding=[1.0, 0.0], auto_embed=False)
    ident = Identity(user_id="anon_parent", student_id="child-a", session_id="scope-a")
    asyncio.run(collect(mod.ProcessUserInput_stream("怎样安排数学练习", identity=ident)))
    prompt = consultant.prompts[-1]
    assert "CHILD_A_ONLY" in prompt
    assert "CHILD_B_ONLY" not in prompt
    assert "FAMILY_ONLY" not in prompt
    assert "LEGACY_SECRET" not in prompt
    mem = next(c for c in mod.get_task_agent(ident.session_id).consultant_agent.context_manager.contexts if isinstance(c, MemoryContext))
    assert mem.identity == ident


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_web_cookies_sessions_ownership_and_untrusted_body(live_chat, path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.routes import router
    from config.identity import IDENTITY_COOKIE_NAME
    mod, svc, identities, _, _ = live_chat
    app = FastAPI()
    app.include_router(router)
    owner = TestClient(app)
    other = TestClient(app)
    first = owner.post(path, json={"message": "初二数学怎么学", "user_id": "default_user"})
    assert first.status_code == 200
    assert "HttpOnly" in first.headers["set-cookie"]
    uid = owner.cookies.get(IDENTITY_COOKIE_NAME)
    assert uid.startswith("anon_") and uid != "default_user"
    sid = first.headers["x-tutor-session-id"]
    assert identities.session_owner(sid) == uid
    second = owner.post(path, json={"message": "继续", "session_id": sid})
    assert second.status_code == 200
    assert len(svc.conversation_repo.get_messages(sid)) == 4
    stolen = other.post(path, json={"message": "窃取会话", "session_id": sid, "user_id": uid})
    assert stolen.status_code == 403
    assert len(svc.conversation_repo.get_messages(sid)) == 4
    assert identities.session_owner(sid) == uid
    new_session = owner.post(path, json={"message": "另一段咨询"})
    assert new_session.status_code == 200
    assert new_session.headers["x-tutor-session-id"] != sid
    assert owner.cookies.get(IDENTITY_COOKIE_NAME) == uid


def test_student_namespace_bound_to_session_not_business_authorization(live_chat):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.routes import router
    mod, svc, _, _, consultant = live_chat
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    a = client.post("/chat", json={"message": "数学方法", "student_id": "business-student-1"})
    assert a.status_code == 200
    sid = a.headers["x-tutor-session-id"]
    switched = client.post("/chat", json={"message": "切换孩子", "session_id": sid, "student_id": "different-child"})
    assert switched.status_code == 409
    assert len(svc.conversation_repo.get_messages(sid)) == 2
    assert "小明" not in consultant.prompts[-1]
    # 未认证namespace只能用于咨询记忆,不能授权已有学员业务读写。
    mod.get_task_agent(sid).task_classifier.llm.responses = ["progress"]
    result = client.post("/chat", json={"message": "查孩子上课记录", "session_id": sid, "student_id": "business-student-1"})
    assert "验证" in result.text or "认证" in result.text


def test_managed_trial_and_recommendation_do_not_double_write(live_chat):
    from config.identity import Identity
    mod, svc, _, _, _ = live_chat
    ident = Identity(user_id="anon_u", session_id="managed")
    # 构建真实business agents后使用它们的共享写方法;不按content去重。
    svc.record_human(ident, "重复")
    agent = mod.get_task_agent(ident.session_id, identity=ident)
    agent.trial_agent._record_shared_ai("中间回复")
    agent.recommendation_agent._record_shared_ai("中间回复")
    assert len(svc.get_history(ident.session_id).messages) == 1
    svc.record_ai(ident, "中间回复")
    svc.record_human(ident, "重复")
    assert [m.content for m in svc.get_history(ident.session_id).messages] == ["重复", "中间回复", "重复"]


def test_cancel_releases_lock_partial_and_next_turn_order(live_chat, monkeypatch):
    from config.identity import Identity
    mod, svc, _, _, _ = live_chat
    ident = Identity(user_id="anon_u", session_id="cancelled")
    ready = asyncio.Event()

    class Slow:
        async def classify_task_stream(self, task):
            yield "[THOUGHT]路由[REPLY]已生成的部分"
            ready.set()
            await asyncio.Future()

    monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Slow())

    async def scenario():
        task = asyncio.create_task(collect(mod.ProcessUserInput_stream("q1", identity=ident)))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        class Done:
            async def classify_task_stream(self, task):
                yield "[REPLY]完成"
        monkeypatch.setattr(mod, "get_task_agent", lambda **kw: Done())
        await collect(mod.ProcessUserInput_stream("q2", identity=ident))
    asyncio.run(scenario())
    rows = svc.conversation_repo.get_messages(ident.session_id)
    assert [r["role"] for r in rows] == ["human", "ai", "human", "ai"]
    assert rows[1]["status"] == "partial"
    assert rows[3]["status"] == "complete"
    assert not mod._active_sessions


def test_drop_and_ttl_never_evict_active_and_can_restore(live_chat, monkeypatch):
    from config.identity import Identity
    mod, svc, _, _, _ = live_chat
    ident = Identity(user_id="anon_u", session_id="cache")
    asyncio.run(collect(mod.ProcessUserInput_stream("数学方法", identity=ident)))
    history = mod.get_session_chat_history(ident.session_id)
    mod._enter_session(ident.session_id)
    mod.drop_session(ident.session_id)
    assert mod.get_session_chat_history(ident.session_id) is history
    assert ident.session_id in mod._session_agents
    mod._exit_session(ident.session_id)
    mod.evict_idle_sessions(now=mod._session_last_access[ident.session_id] + mod.SESSION_CACHE_TTL_SECONDS + 1)
    assert ident.session_id not in mod._session_agents
    assert ident.session_id not in mod._session_chat_histories
    assert ident.session_id not in mod._session_async_locks
    asyncio.run(collect(mod.ProcessUserInput_stream("继续数学咨询", identity=ident)))
    assert len(mod.get_session_chat_history(ident.session_id).messages) == 4
