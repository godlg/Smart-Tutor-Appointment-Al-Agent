"""Web 身份入口回归：真实临时 repository，仅替换聊天生成边界。"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import event

from api import identity as identity_api
from config.identity import (
    IDENTITY_COOKIE_NAME,
    Identity,
    get_current_identity,
    reset_current_identity,
    set_current_identity,
)
from db.models import SessionMessage
from services.identity_service import IdentityService
from web import routes


@dataclass(frozen=True)
class WebEntry:
    service: IdentityService
    app: FastAPI
    client: TestClient
    calls: list
    db_url: str


@pytest.fixture(params=["/chat", "/chat/stream"])
def chat_path(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_url = f"sqlite:///{tmp_path / 'web-entry.db'}"
    service = IdentityService(db_url)
    calls = []
    monkeypatch.setattr(identity_api, "_identity_service", service)
    monkeypatch.delenv("TUTOR_IDENTITY_COOKIE_SECURE", raising=False)

    async def fake_stream(message: str, session_id: str | None = None):
        ident = get_current_identity()
        binding = service.repo.get_session(ident.session_id) if ident else None
        calls.append((message, session_id, ident, binding))
        yield "[REPLY]离线回复"

    monkeypatch.setattr(routes, "ProcessUserInput_stream", fake_stream)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield WebEntry(service, app, client, calls, db_url)
    service.close()


def identity_cookie(response) -> object:
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return cookie[IDENTITY_COOKIE_NAME]


def forbid_stream_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("身份/归属失败不能先构造 StreamingResponse")
    monkeypatch.setattr(routes, "StreamingResponse", forbidden)


def test_new_session_is_bound_before_stream_and_cookie_is_persistent(entry: WebEntry, chat_path: str):
    response = entry.client.post(chat_path, json={"message": "你好", "user_id": "default_user"})
    assert response.status_code == 200
    sid = response.headers["x-tutor-session-id"]
    uid = entry.client.cookies.get(IDENTITY_COOKIE_NAME)
    assert sid.startswith("sess_") and len(sid) > 20
    assert uid.startswith("anon_") and uid != "default_user"
    assert entry.service.session_owner(sid) == uid
    _, passed_sid, ident, binding = entry.calls[-1]
    assert passed_sid is None  # 保留旧入口 kwargs;实际会话由完整 ContextVar 贯穿。
    assert ident == Identity(user_id=uid, session_id=sid)
    assert binding == {"session_id": sid, "user_id": uid, "student_id": None}
    cookie = identity_cookie(response)
    assert cookie["httponly"] and cookie["samesite"].lower() == "lax"
    assert cookie["path"] == "/" and int(cookie["max-age"]) > 86400
    assert cookie["expires"] and not cookie["secure"]
    assert get_current_identity() is None


def test_existing_frontend_session_id_is_preserved(entry: WebEntry, chat_path: str):
    response = entry.client.post(chat_path, json={"message": "你好", "session_id": "browser-existing-uuid"})
    assert response.status_code == 200
    assert response.headers["x-tutor-session-id"] == "browser-existing-uuid"
    assert entry.calls[-1][1] == "browser-existing-uuid"
    assert entry.calls[-1][2].session_id == "browser-existing-uuid"
    assert entry.calls[-1][3]["user_id"] == entry.client.cookies.get(IDENTITY_COOKIE_NAME)


def test_same_cookie_reuses_owner_but_no_session_starts_a_new_session(entry: WebEntry, chat_path: str):
    first = entry.client.post(chat_path, json={"message": "第一段"})
    first_sid = first.headers["x-tutor-session-id"]
    uid = entry.client.cookies.get(IDENTITY_COOKIE_NAME)
    second = entry.client.post(chat_path, json={"message": "继续", "session_id": first_sid})
    third = entry.client.post(chat_path, json={"message": "新会话"})
    assert second.headers["x-tutor-session-id"] == first_sid
    assert third.headers["x-tutor-session-id"] != first_sid
    assert entry.client.cookies.get(IDENTITY_COOKIE_NAME) == uid
    assert len(entry.service.repo.list_sessions_for_user(uid)) == 2


def test_owner_and_student_binding_survive_service_rebuild(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch):
    first = entry.client.post(chat_path, json={"message": "第一段", "student_id": "child-a"})
    sid = first.headers["x-tutor-session-id"]
    uid = entry.client.cookies.get(IDENTITY_COOKIE_NAME)
    rebuilt = IdentityService(entry.db_url)
    monkeypatch.setattr(identity_api, "_identity_service", rebuilt)
    try:
        again = entry.client.post(chat_path, json={"message": "继续", "session_id": sid})
        assert again.status_code == 200
        assert entry.calls[-1][2] == Identity(user_id=uid, student_id="child-a", session_id=sid)
        with TestClient(entry.app) as other:
            denied = other.post(chat_path, json={"message": "冒充", "session_id": sid, "user_id": uid})
        assert denied.status_code == 403
        assert rebuilt.session_owner(sid) == uid
    finally:
        rebuilt.close()


def test_student_namespace_never_queries_business_students(entry: WebEntry, chat_path: str):
    def reject_business_student_query(conn, cursor, statement, parameters, context, executemany):
        assert "student_profiles" not in statement.lower(), "namespace不证明业务学员归属"
    engine = entry.service.db_router.session_manager.engine
    event.listen(engine, "before_cursor_execute", reject_business_student_query)
    try:
        family = entry.client.post(chat_path, json={"message": "家庭"})
        child_a = entry.client.post(chat_path, json={"message": "孩子甲", "student_id": "business-student-1"})
        child_b = entry.client.post(chat_path, json={"message": "孩子乙", "student_id": "child-b"})
    finally:
        event.remove(engine, "before_cursor_execute", reject_business_student_query)
    assert all(response.status_code == 200 for response in (family, child_a, child_b))
    identities = [call[2] for call in entry.calls]
    assert [ident.student_id for ident in identities] == [None, "business-student-1", "child-b"]
    assert len({ident.memory_scope() for ident in identities}) == 3
    assert len({ident.user_id for ident in identities}) == 1
    for ident in identities:
        assert entry.service.repo.get_session(ident.session_id)["student_id"] == ident.student_id


@pytest.mark.parametrize("student_id", ["child-b", None])
def test_student_switch_including_explicit_null_is_409_before_stream(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch, student_id: str | None):
    response = entry.client.post(chat_path, json={"message": "孩子甲", "student_id": "child-a"})
    sid = response.headers["x-tutor-session-id"]
    forbid_stream_creation(monkeypatch)
    denied = entry.client.post(chat_path, json={"message": "切换", "session_id": sid, "student_id": student_id})
    assert denied.status_code == 409
    assert "x-tutor-session-id" not in denied.headers
    assert len(entry.calls) == 1
    assert entry.service.repo.get_session(sid)["student_id"] == "child-a"


def test_other_owner_cannot_claim_student_namespace_before_stream(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch):
    owner = entry.service.issue_new_identity()
    entry.service.claim_student(owner, "claimed-child")
    forbid_stream_creation(monkeypatch)
    response = entry.client.post(chat_path, json={"message": "借用孩子", "student_id": "claimed-child", "user_id": owner.user_id})
    assert response.status_code == 409
    assert entry.calls == []
    uid = entry.client.cookies.get(IDENTITY_COOKIE_NAME)
    assert uid != owner.user_id
    assert entry.service.repo.list_sessions_for_user(uid) == []
    assert entry.service.repo.get_student_owner("claimed-child") == owner.user_id


def test_forged_cookie_and_body_user_id_cannot_authorize_foreign_session(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch):
    owner = entry.service.issue_new_identity()
    sid = entry.service.create_session(owner)
    entry.client.cookies.set(IDENTITY_COOKIE_NAME, "default_user")
    forbid_stream_creation(monkeypatch)
    response = entry.client.post(chat_path, json={"message": "冒充", "session_id": sid, "user_id": owner.user_id})
    assert response.status_code == 403
    assert "x-tutor-session-id" not in response.headers
    assert entry.calls == []
    assert entry.service.session_owner(sid) == owner.user_id
    cookie = identity_cookie(response)
    assert cookie.value.startswith("anon_") and cookie.value != owner.user_id
    assert cookie["httponly"] and int(cookie["max-age"]) > 0


@pytest.mark.parametrize("has_binding, expected_status", [(False, 403), (True, 409)])
def test_unknown_or_inconsistent_archive_is_rejected_before_stream(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch, has_binding: bool, expected_status: int):
    ident = entry.service.issue_new_identity()
    sid = "archived-session"
    if has_binding:
        entry.service.create_session(ident, sid)
    with entry.service.db_router.session_manager.session_scope() as session:
        session.add(SessionMessage(session_id=sid, user_id=ident.user_id, student_id="unexpected-child", seq=1, role="human", content="PRIVATE"))
    entry.client.cookies.set(IDENTITY_COOKIE_NAME, ident.user_id)
    forbid_stream_creation(monkeypatch)
    response = entry.client.post(chat_path, json={"message": "读档", "session_id": sid})
    assert response.status_code == expected_status
    assert "PRIVATE" not in response.text
    assert entry.calls == []


@pytest.mark.parametrize("scheme, force_secure, expected", [("http", "false", False), ("https", "false", True), ("http", "true", True)])
def test_cookie_secure_defaults_to_https_and_can_be_forced_for_proxy(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch, scheme: str, force_secure: str, expected: bool):
    monkeypatch.setenv("TUTOR_IDENTITY_COOKIE_SECURE", force_secure)
    with TestClient(entry.app, base_url=f"{scheme}://testserver") as client:
        response = client.post(chat_path, json={"message": "安全cookie"})
    assert response.status_code == 200
    assert bool(identity_cookie(response)["secure"]) is expected


def test_contextvar_is_reset_after_stream_error(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch):
    async def failing_stream(message: str, session_id: str | None = None):
        ident = get_current_identity()
        assert ident is not None and ident.session_id
        assert entry.service.session_owner(ident.session_id) == ident.user_id
        yield "部分回复"
        raise RuntimeError("offline generation failed")
    monkeypatch.setattr(routes, "ProcessUserInput_stream", failing_stream)

    async def scenario() -> None:
        outer = Identity(user_id="outer-scope")
        token = set_current_identity(outer)
        try:
            request = Request({"type": "http", "scheme": "http", "server": ("testserver", 80), "path": chat_path, "headers": []})
            endpoint = routes.chat_endpoint if chat_path == "/chat" else routes.chat_stream_endpoint
            response = await endpoint(routes.ChatRequest(message="问题"), request)
            with pytest.raises(RuntimeError, match="offline generation failed"):
                async for _ in response.body_iterator:
                    pass
            assert get_current_identity() == outer
        finally:
            reset_current_identity(token)
    asyncio.run(scenario())
    assert get_current_identity() is None


@pytest.mark.parametrize("unsafe_sid", ["会话一", "bad\r\nX-Test: injected", "bad\n", "bad\x00", "x" * 257])
def test_unsafe_header_session_id_is_rejected_without_binding(entry: WebEntry, chat_path: str, unsafe_sid: str):
    response = entry.client.post(chat_path, json={"message": "问题", "session_id": unsafe_sid})
    assert response.status_code == 422
    assert "set-cookie" not in response.headers
    assert entry.service.repo.get_session(unsafe_sid) is None
    assert entry.calls == []


def test_sync_identity_transaction_is_not_run_on_event_loop(entry: WebEntry, chat_path: str, monkeypatch: pytest.MonkeyPatch):
    original_assert = entry.service.assert_session_owned
    loop_thread_id = threading.get_ident()
    observed_threads = []

    def inspect_thread(identity: Identity, session_id: str) -> str:
        current_thread = threading.get_ident()
        observed_threads.append(current_thread)
        assert current_thread != loop_thread_id, "同步SQLite绑定/锁等待不能阻塞事件循环"
        return original_assert(identity, session_id)
    monkeypatch.setattr(entry.service, "assert_session_owned", inspect_thread)

    async def scenario() -> None:
        request = Request({"type": "http", "scheme": "http", "server": ("testserver", 80), "path": chat_path, "headers": []})
        endpoint = routes.chat_endpoint if chat_path == "/chat" else routes.chat_stream_endpoint
        response = await endpoint(routes.ChatRequest(message="问题"), request)
        parts = [part async for part in response.body_iterator]
        assert parts == ["[REPLY]离线回复"]
        assert response.headers["x-tutor-session-id"] == entry.calls[-1][2].session_id
        assert get_current_identity() is None
    asyncio.run(scenario())
    assert observed_threads and all(t != loop_thread_id for t in observed_threads)


@pytest.fixture
def real_process(entry: WebEntry, monkeypatch: pytest.MonkeyPatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from api import chat_handler
    from agents import consultant_agent, task_classification_agent, recommendation_agent
    from agents.trial import trial_agent
    from agents.lesson import lesson_agent
    from services import text_embedding, knowledge_service, hybrid_retriever
    from services.conversation_service import ConversationService

    conversation = ConversationService(entry.db_url)
    classifier_model = FakeListChatModel(responses=["consultation"])
    consultant_model = FakeListChatModel(responses=["YES", "离线数学学习建议。"])
    monkeypatch.setattr(consultant_agent.ConsultantAgent, "_initialize_llm", lambda self: consultant_model)
    for module in (task_classification_agent, recommendation_agent, trial_agent, lesson_agent):
        monkeypatch.setattr(module, "create_chat_model", lambda **kwargs: classifier_model)
    for module in (text_embedding, knowledge_service, hybrid_retriever):
        monkeypatch.setattr(module, "embed_input", lambda text: [1.0, 0.0])
    monkeypatch.setattr(chat_handler, "_conversation_service", conversation)
    for name in ("_session_agents", "_session_chat_histories", "_session_async_locks", "_active_sessions", "_session_last_access"):
        monkeypatch.setattr(chat_handler, name, {}, raising=False)
    monkeypatch.setattr(chat_handler, "_task_agent", None)
    monkeypatch.setattr(routes, "ProcessUserInput_stream", chat_handler.ProcessUserInput_stream)
    yield chat_handler, conversation
    conversation.close()


def test_real_process_uses_context_session_for_two_browsers_and_resumes_student(entry: WebEntry, real_process, chat_path: str):
    mod, conversation = real_process
    first = entry.client.post(chat_path, json={"message": "初二数学方法", "student_id": "child-a"})
    assert first.status_code == 200
    sid_a = first.headers["x-tutor-session-id"]
    uid_a = entry.client.cookies.get(IDENTITY_COOKIE_NAME)
    with TestClient(entry.app) as other:
        second = other.post(chat_path, json={"message": "初中数学预习"})
        assert second.status_code == 200
        sid_b = second.headers["x-tutor-session-id"]
        uid_b = other.cookies.get(IDENTITY_COOKIE_NAME)
    assert sid_a != sid_b and uid_a != uid_b
    assert mod._session_agents[sid_a] is not mod._session_agents[sid_b]
    assert conversation.get_history(sid_a) is not conversation.get_history(sid_b)
    assert mod._task_agent is None  # 两次缺客户端sid的HTTP请求都未走global。
    again = entry.client.post(chat_path, json={"message": "继续数学练习", "session_id": sid_a})
    assert again.status_code == 200
    assert again.headers["x-tutor-session-id"] == sid_a
    with conversation.db_router.session_manager.session_scope() as session:
        a_rows = session.query(SessionMessage).filter_by(session_id=sid_a).order_by(SessionMessage.seq).all()
        b_rows = session.query(SessionMessage).filter_by(session_id=sid_b).order_by(SessionMessage.seq).all()
        assert [row.role for row in a_rows] == ["human", "ai", "human", "ai"]
        assert [row.role for row in b_rows] == ["human", "ai"]
        assert all(row.user_id == uid_a and row.student_id == "child-a" for row in a_rows)
        assert all(row.user_id == uid_b and row.student_id is None for row in b_rows)
        assert all("[THOUGHT]" not in row.content for row in a_rows + b_rows)
    assert get_current_identity() is None
