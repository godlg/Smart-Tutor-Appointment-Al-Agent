"""家长侧轮询端点 `GET /api/handoffs/session/{id}/messages` 的会话归属校验（潜在 IDOR 修复）。

- 全部走临时文件库（显式传 url），不构造默认库的服务。
- 未绑定 session 仍按既有语义 200 + []（首次出现即绑定给当前身份）。
"""

import pytest

SECRET_PARENT = "SECRET-PARENT-MSG-7f3a"
SECRET_AGENT = "SECRET-AGENT-REPLY-9c1d"


@pytest.fixture
def env(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from api import identity as api_identity
    from services.human_handoff_service import HumanHandoffService
    from services.identity_service import IdentityService

    url = f"sqlite:///{(tmp_path / 'session_ownership.db').as_posix()}"
    identities = IdentityService(url)

    def _patched_init(self, db_path_=None, threshold=None):
        self.db_router = identities.db_router
        self.repo = self.db_router.handoff_repo
        self.threshold = self._resolve_threshold(threshold)
        self._kickback_counts = {}
        self._handed_off_sessions = set()
        self._recent_inputs = {}

    monkeypatch.setattr(HumanHandoffService, "__init__", _patched_init)
    monkeypatch.setattr(api_identity, "_identity_service", identities)
    yield TestClient(app), identities, HumanHandoffService()
    identities.close()


def _bind_victim_session(identities, handoff, session_id, owner_user_id):
    """把 session 绑定给 owner，并写入一条工单 + 家长消息 + 客服回复。"""
    from config.identity import Identity

    owner = Identity(user_id=owner_user_id, legacy=False)
    identities.create_session(owner, session_id)
    record = handoff.trigger_handoff(session_id, "keyword", reason="人工")
    handoff.parent_message(session_id, SECRET_PARENT)
    handoff.claim(record["id"], "客服小美")
    handoff.agent_reply(record["id"], "客服小美", SECRET_AGENT)


def _poll(client, session_id, user_id=None):
    from config.identity import IDENTITY_COOKIE_NAME

    client.cookies.clear()
    if user_id is not None:
        client.cookies.set(IDENTITY_COOKIE_NAME, user_id)
    return client.get(f"/api/handoffs/session/{session_id}/messages")


def test_other_identity_gets_403_without_message_content(env):
    client, identities, handoff = env
    owner = identities.issue_new_identity()
    intruder = identities.issue_new_identity()
    _bind_victim_session(identities, handoff, "victim-session", owner.user_id)

    r = _poll(client, "victim-session", intruder.user_id)

    assert r.status_code == 403
    assert SECRET_PARENT not in r.text
    assert SECRET_AGENT not in r.text
    assert r.json() == {"detail": "该会话不属于当前身份"}


def test_cookieless_request_to_bound_session_is_403(env):
    client, identities, handoff = env
    owner = identities.issue_new_identity()
    _bind_victim_session(identities, handoff, "victim-session", owner.user_id)

    r = _poll(client, "victim-session")

    assert r.status_code == 403
    assert SECRET_PARENT not in r.text
    assert SECRET_AGENT not in r.text


def _count_identity_rows(identities):
    from db.models import IdentityRecord

    with identities.db_router.identity_repo.session_manager.session_scope() as session:
        return session.query(IdentityRecord).count()


def test_cookieless_request_does_not_issue_identity_or_write_db(env):
    client, identities, handoff = env
    owner = identities.issue_new_identity()
    _bind_victim_session(identities, handoff, "victim-session", owner.user_id)
    before = _count_identity_rows(identities)

    r = _poll(client, "victim-session")

    assert r.status_code == 403
    assert _count_identity_rows(identities) == before


def test_owner_gets_200_with_parent_and_agent_messages(env):
    client, identities, handoff = env
    owner = identities.issue_new_identity()
    _bind_victim_session(identities, handoff, "victim-session", owner.user_id)

    r = _poll(client, "victim-session", owner.user_id)

    assert r.status_code == 200, r.text
    assert [m["role"] for m in r.json()["data"]] == ["parent", "agent"]


def test_unbound_session_keeps_200_empty_and_binds_to_caller(env):
    client, identities, _ = env
    owner = identities.issue_new_identity()

    r = _poll(client, "fresh-session", owner.user_id)

    assert r.status_code == 200, r.text
    assert r.json()["data"] == []
    assert identities.session_owner("fresh-session") == owner.user_id


@pytest.mark.parametrize("path", ["/api/handoffs/session//messages"])
def test_empty_session_id_keeps_framework_behavior(env, path):
    client, _, _ = env

    r = client.get(path)

    assert r.status_code in (404, 422), r.text
