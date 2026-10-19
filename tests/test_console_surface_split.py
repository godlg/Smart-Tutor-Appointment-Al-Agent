"""控制台面拆分验收测试（机构后台分面 B）

覆盖：
1. 面矩阵（本 spec 核心）：后台页 401 / 角色匹配 200 / 角色越权 403；登录页公开
2. 旧路径已下线：`/knowledge`、`/human-handoff`、`/user_behavior_analysis` → 404
3. 家长面不受影响：无 cookie 下家长页全部 200
4. 后台 API 闸门：`/api/knowledge/*`、客服侧 `/api/handoffs/*` 未登录 401
5. C 端未被误伤：家长侧 `GET /api/handoffs/session/{id}/messages` 无 cookie → 200

约定：不用 pytest-asyncio；FastAPI 路由用 TestClient；直连服务一律显式传临时库
（`tmp_path` 文件库，不用 :memory:，TestClient 在独立线程跑 app）。
"""

from __future__ import annotations

import pytest

from config.console_auth import CONSOLE_COOKIE_NAME
from config.identity import IDENTITY_COOKIE_NAME

# 后台页 → (只允许的角色, 被拒绝的角色)
PROTECTED_PAGES = [
    ("/console/knowledge", "academic", "support"),
    ("/console/human-handoff", "support", "academic"),
]


def _db_url(tmp_path, name: str) -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """TestClient + 控制台鉴权服务指向临时文件库 + 人工接管服务指向临时文件库。

    控制台账号：把 `ConsoleAuthService.__init__` 的默认 db_path 换成本用例的临时
    文件库（`/api/console/login` 是自行 `ConsoleAuthService()` 构造的），并清空
    依赖里的模块级单例（`api/core/console_auth.py`），避免跨用例串库。

    人工接管：`api/handoff.py` 端点自行构造 `HumanHandoffService()`，同样把默认
    db_path 指向本用例临时库，便于造一条「已知存在的」工单来验 resolve 闸门。
    """
    from fastapi.testclient import TestClient
    from api.core import console_auth as console_dep
    from app import app
    from db.db_router import DatabaseRouter
    from services.console_auth_service import ConsoleAuthService
    from services.human_handoff_service import HumanHandoffService

    console_db = _db_url(tmp_path, "console_split.db")
    handoff_db = _db_url(tmp_path, "handoff_split.db")

    original_console_init = ConsoleAuthService.__init__

    def _patched_console_init(self, db_path=None, *args, **kwargs):
        original_console_init(self, db_path=db_path or console_db)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_console_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)

    shared = {"router": None}

    def _patched_handoff_init(self, db_path_=None, threshold=None):
        if shared["router"] is None:
            shared["router"] = DatabaseRouter(handoff_db)
        self.db_router = shared["router"]
        self.repo = self.db_router.handoff_repo
        self.threshold = self._resolve_threshold(threshold)
        self._kickback_counts = {}
        self._handed_off_sessions = set()

    monkeypatch.setattr(HumanHandoffService, "__init__", _patched_handoff_init)

    ConsoleAuthService(db_path=console_db).ensure_default_users()

    try:
        yield TestClient(app)
    finally:
        if shared["router"] is not None:
            shared["router"].close()


@pytest.fixture
def identity_service(monkeypatch, tmp_path):
    """匿名身份服务指向临时文件库：家长侧端点的 cookie 校验走 `api.identity` 单例，不能落到默认（生产）库。"""
    from api import identity as api_identity
    from services.identity_service import IdentityService

    service = IdentityService(_db_url(tmp_path, "identity_split.db"))
    monkeypatch.setattr(api_identity, "_identity_service", service)
    yield service
    service.close()


def _login(client, username: str, password: str):
    """登录并断言 cookie 已签发；返回响应。"""
    client.cookies.clear()
    response = client.post(
        "/api/console/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    assert CONSOLE_COOKIE_NAME in response.cookies
    return response


# ----------------------------------------------------------------------
# 1. 面矩阵（越权验收）
# ----------------------------------------------------------------------

class TestConsoleSurfaceMatrix:

    @pytest.mark.parametrize("path", [p for p, _, _ in PROTECTED_PAGES] + ["/console"])
    def test_protected_pages_without_cookie_are_401(self, app_env, path):
        client = app_env
        client.cookies.clear()
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path,allowed,denied", PROTECTED_PAGES)
    def test_allowed_role_is_200(self, app_env, path, allowed, denied):
        client = app_env
        _login(client, allowed, f"{allowed}12345")
        r = client.get(path)
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")

    @pytest.mark.parametrize("path,allowed,denied", PROTECTED_PAGES)
    def test_mismatched_role_is_403(self, app_env, path, allowed, denied):
        client = app_env
        _login(client, denied, f"{denied}12345")
        assert client.get(path).status_code == 403

    @pytest.mark.parametrize("username", ["admin", "support", "academic"])
    def test_console_home_allows_console_roles(self, app_env, username):
        client = app_env
        _login(client, username, f"{username}12345")
        r = client.get("/console")
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")

    def test_console_home_lists_three_entries(self, app_env):
        client = app_env
        _login(client, "admin", "admin12345")
        body = client.get("/console").text
        for href in ("/console/human-handoff", "/console/knowledge",
                     "/console/user_behavior_analysis"):
            assert href in body, f"后台首页应列出入口: {href}"
        assert "/api/console/me" in body, "后台首页应回显当前账号"

    def test_login_page_is_public(self, app_env):
        client = app_env
        client.cookies.clear()
        r = client.get("/console/login")
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")
        assert "/api/console/login" in r.text


# ----------------------------------------------------------------------
# 2. 旧路径已下线（不留重定向，避免未鉴权入口）
# ----------------------------------------------------------------------

class TestLegacyPathsRetired:

    @pytest.mark.parametrize("path", [
        "/knowledge", "/human-handoff", "/user_behavior_analysis",
    ])
    def test_legacy_path_is_404(self, app_env, path):
        client = app_env
        client.cookies.clear()
        assert client.get(path).status_code == 404

    @pytest.mark.parametrize("path", [
        "/knowledge", "/human-handoff", "/user_behavior_analysis",
    ])
    def test_legacy_path_is_404_even_when_logged_in(self, app_env, path):
        """登录也不该把旧路径「请回来」——旧入口必须彻底不存在。"""
        client = app_env
        _login(client, "admin", "admin12345")
        assert client.get(path).status_code == 404


# ----------------------------------------------------------------------
# 3. 家长面不受影响（绝不加鉴权）
# ----------------------------------------------------------------------

class TestParentSurfaceUnaffected:

    @pytest.mark.parametrize("path", [
        "/", "/teachers", "/packages", "/parent-center",
        "/schedule-calendar", "/progress-dashboard",
        "/lesson-series", "/student-profile", "/teachers/1",
    ])
    def test_parent_pages_200_without_cookie(self, app_env, path):
        client = app_env
        client.cookies.clear()
        r = client.get(path)
        assert r.status_code == 200, f"{path} 家长面不应被鉴权影响: {r.text[:200]}"


# ----------------------------------------------------------------------
# 4. 后台 API 闸门（路径不变，只加 dependencies）
# ----------------------------------------------------------------------

class TestConsoleApiGates:

    def test_knowledge_list_without_cookie_is_401(self, app_env):
        client = app_env
        client.cookies.clear()
        assert client.get("/api/knowledge/").status_code == 401

    def test_knowledge_list_with_academic_is_200(self, app_env):
        client = app_env
        _login(client, "academic", "academic12345")
        r = client.get("/api/knowledge/")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "success"

    def test_knowledge_list_with_support_is_403(self, app_env):
        client = app_env
        _login(client, "support", "support12345")
        assert client.get("/api/knowledge/").status_code == 403

    @pytest.fixture
    def known_handoff_id(self, app_env):
        """造一条真实存在的 pending 工单，供闸门用例使用。"""
        from services.human_handoff_service import HumanHandoffService

        svc = HumanHandoffService()
        record = svc.trigger_handoff("split-gate-sess", "keyword", reason="闸门用例")
        return record["id"]

    def test_handoff_resolve_without_cookie_is_401(self, app_env, known_handoff_id):
        client = app_env
        client.cookies.clear()
        r = client.post(f"/api/handoffs/{known_handoff_id}/resolve", json={})
        assert r.status_code == 401

    def test_handoff_resolve_with_support_is_not_401(self, app_env, known_handoff_id):
        client = app_env
        _login(client, "support", "support12345")
        r = client.post(f"/api/handoffs/{known_handoff_id}/resolve",
                        json={"handled_by": "客服小美"})
        assert r.status_code != 401, r.text
        assert r.status_code == 200, r.text
        assert r.json()["data"]["status"] == "resolved"

    def test_handoff_list_without_cookie_is_401(self, app_env):
        client = app_env
        client.cookies.clear()
        assert client.get("/api/handoffs?status=pending").status_code == 401


# ----------------------------------------------------------------------
# 5. C 端未被误伤：家长侧轮询端点不带控制台依赖
# ----------------------------------------------------------------------

class TestParentSideEndpointNotGated:

    @pytest.mark.parametrize("session_id", ["any-session", "split-gate-sess", "nobody"])
    def test_session_messages_200_with_identity_cookie_no_console_role(
        self, app_env, identity_service, session_id
    ):
        """带匿名身份 cookie、且不带任何控制台 cookie → 200：证明未被控制台闸门误伤。

        这些 session 在临时库里都未绑定，按既有语义 200 + []。
        """
        client = app_env
        client.cookies.clear()
        owner = identity_service.issue_new_identity()
        client.cookies.set(IDENTITY_COOKIE_NAME, owner.user_id)
        assert CONSOLE_COOKIE_NAME not in client.cookies
        r = client.get(f"/api/handoffs/session/{session_id}/messages")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []

    def test_session_messages_without_console_role_still_200(self, app_env, identity_service):
        """即使登录了无权访问客服侧的角色，家长侧端点也照常可读（它是 C 端接口）。"""
        client = app_env
        _login(client, "academic", "academic12345")
        owner = identity_service.issue_new_identity()
        client.cookies.set(IDENTITY_COOKIE_NAME, owner.user_id)
        r = client.get("/api/handoffs/session/any-session/messages")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []

    def test_session_messages_without_identity_cookie_is_403_not_console_gate(self, app_env):
        """无身份 cookie → 403，且 403 来自会话归属校验（detail 为「该会话不属于当前身份」），
        不是控制台闸门（控制台闸门对未登录返回 401）。防止新契约将来被无声回退成 200。
        """
        client = app_env
        client.cookies.clear()
        r = client.get("/api/handoffs/session/any-session/messages")
        assert r.status_code == 403, r.text
        assert r.json() == {"detail": "该会话不属于当前身份"}
