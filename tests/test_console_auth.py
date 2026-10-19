"""控制台账号与鉴权基座测试(控制台分面 A)

覆盖:
- PBKDF2 哈希/校验(随机盐、往返、篡改、非法格式)
- 存储格式断言
- ensure_default_users 幂等(按 username 查缺补漏)
- 三个默认账号登录 / 失败路径 / is_active=False
- resolve 有效/伪造/过期(过期行被删)/logout 后失效
- 依赖矩阵 401/401/200/403/200(本 spec 核心验收)
- 端到端 login → me → logout → me

约定:不用 pytest-asyncio;FastAPI 路由用 TestClient;直连服务用 tmp_path 文件库
(不用 :memory:,TestClient 在独立线程跑 app)。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.console_auth import CONSOLE_COOKIE_NAME
from db.models import ConsoleSession, ConsoleUser
from services.console_auth_service import (
    AuthError,
    ConsoleAuthService,
    hash_password,
    verify_password,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

def _db_url(tmp_path, name="console_auth.db") -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


@pytest.fixture
def db_path(tmp_path):
    return _db_url(tmp_path)


@pytest.fixture
def service(db_path):
    return ConsoleAuthService(db_path=db_path)


@pytest.fixture
def seeded_service(service):
    service.ensure_default_users()
    return service


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient + 把控制台鉴权服务默认指向临时文件库。

    路由是「按需 import 后自行 ConsoleAuthService()」的,所以把 __init__ 的默认
    db_path 换成本用例的临时文件库(与 conftest 对 IdentityService 的做法同形);
    依赖里的模块级单例也一并清空,避免跨用例串库。
    """
    from fastapi.testclient import TestClient
    from app import app
    from api.core import console_auth as console_dep

    db_path = _db_url(tmp_path, "console_api.db")
    original_init = ConsoleAuthService.__init__

    def _patched_init(self, db_path_=None, *args, **kwargs):
        original_init(self, db_path=db_path_ or db_path)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)

    ConsoleAuthService(db_path=db_path).ensure_default_users()
    yield TestClient(app)


# ----------------------------------------------------------------------
# 1. 哈希与格式
# ----------------------------------------------------------------------

class TestPasswordHashing:

    def test_same_password_different_salt(self):
        first = hash_password("admin12345")
        second = hash_password("admin12345")
        assert first != second
        assert verify_password(first, "admin12345")
        assert verify_password(second, "admin12345")

    def test_roundtrip_true_and_wrong_password_false(self):
        stored = hash_password("support12345")
        assert verify_password(stored, "support12345") is True
        assert verify_password(stored, "support12346") is False
        assert verify_password(stored, "") is False

    def test_tampered_digest_false(self):
        stored = hash_password("academic12345")
        head, digest = stored.rsplit("$", 1)
        flipped = "0" if digest[0] != "0" else "1"
        tampered = f"{head}${flipped}{digest[1:]}"
        assert tampered != stored
        assert verify_password(tampered, "academic12345") is False

    @pytest.mark.parametrize("bad", [
        "",
        "not-a-hash",
        "pbkdf2_sha256$120000$deadbeef",
        "pbkdf2_sha256$120000$zz$zz",
        "pbkdf2_sha256$0$aabb$ccdd",
        "md5$120000$aabb$ccdd",
        "pbkdf2_sha256$abc$aabb$ccdd",
        None,
    ])
    def test_invalid_format_returns_false_without_raising(self, bad):
        assert verify_password(bad, "whatever") is False

    def test_stored_format_prefix_and_four_segments(self):
        stored = hash_password("admin12345")
        assert stored.startswith("pbkdf2_sha256$120000$")
        parts = stored.split("$")
        assert len(parts) == 4
        assert parts[0] == "pbkdf2_sha256"
        assert parts[1] == "120000"
        assert len(bytes.fromhex(parts[2])) == 16
        assert len(bytes.fromhex(parts[3])) == 32


# ----------------------------------------------------------------------
# 2. 默认账号种子(幂等 / 查缺补漏)
# ----------------------------------------------------------------------

class TestEnsureDefaultUsers:

    def test_idempotent_second_call_inserts_zero(self, service):
        assert service.ensure_default_users() == 3
        assert service.ensure_default_users() == 0
        with service.db.session_scope() as session:
            assert session.query(ConsoleUser).count() == 3

    def test_restores_only_missing_user(self, service):
        service.ensure_default_users()
        with service.db.session_scope() as session:
            before = {
                (row.username, row.password_hash, row.role, row.display_name)
                for row in session.query(ConsoleUser).all()
            }

        with service.db.session_scope() as session:
            target = session.query(ConsoleUser).filter(ConsoleUser.username == "support").one()
            session.delete(target)

        with service.db.session_scope() as session:
            assert session.query(ConsoleUser).count() == 2

        assert service.ensure_default_users() == 1

        with service.db.session_scope() as session:
            after = {
                (row.username, row.password_hash, row.role, row.display_name)
                for row in session.query(ConsoleUser).all()
            }
        assert {row[0] for row in after} == {"admin", "support", "academic"}
        # 其余两行未被重写(哈希与角色原样保留),唯一变化是补回的 support
        untouched = {row for row in before if row[0] != "support"}
        assert untouched.issubset(after)
        assert after - untouched == {row for row in after if row[0] == "support"}

    def test_default_roles_and_display_names(self, seeded_service):
        with seeded_service.db.session_scope() as session:
            rows = {
                row.username: (row.role, row.display_name, row.is_active)
                for row in session.query(ConsoleUser).all()
            }
        assert set(rows) == {"admin", "support", "academic"}
        assert rows["admin"][0] == "admin"
        assert rows["support"][0] == "support"
        assert rows["academic"][0] == "academic"
        assert rows["admin"][1] == "教务主管"
        assert rows["admin"][2] is True


# ----------------------------------------------------------------------
# 3. 登录
# ----------------------------------------------------------------------

class TestLogin:

    @pytest.mark.parametrize("username,password,role", [
        ("admin", "admin12345", "admin"),
        ("support", "support12345", "support"),
        ("academic", "academic12345", "academic"),
    ])
    def test_default_accounts_login(self, seeded_service, username, password, role):
        result = seeded_service.login(username, password)
        assert result["username"] == username
        assert result["role"] == role
        assert result["display_name"]
        assert isinstance(result["token"], str) and 0 < len(result["token"]) <= 64
        assert datetime.fromisoformat(result["expires_at"]) > datetime.now()
        assert "password_hash" not in result

    def test_wrong_password_raises_auth_error(self, seeded_service):
        with pytest.raises(AuthError):
            seeded_service.login("admin", "wrong-password")

    def test_unknown_user_raises_same_auth_error(self, seeded_service):
        with pytest.raises(AuthError) as exc:
            seeded_service.login("ghost", "whatever")
        assert str(exc.value) == "用户名或密码错误"

    def test_inactive_user_cannot_login(self, service):
        service.ensure_default_users()
        with service.db.session_scope() as session:
            row = session.query(ConsoleUser).filter(ConsoleUser.username == "support").one()
            row.is_active = False
        with pytest.raises(AuthError) as exc:
            service.login("support", "support12345")
        assert str(exc.value) == "用户名或密码错误"


# ----------------------------------------------------------------------
# 4. resolve / logout
# ----------------------------------------------------------------------

class TestResolve:

    def test_valid_token_returns_user(self, seeded_service):
        issued = seeded_service.login("admin", "admin12345")
        user = seeded_service.resolve(issued["token"])
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"
        assert user["display_name"] == "教务主管"
        assert isinstance(user["user_id"], int)
        assert "password_hash" not in user

    def test_forged_token_returns_none(self, seeded_service):
        assert seeded_service.resolve("forged-token-not-issued") is None
        assert seeded_service.resolve(None) is None
        assert seeded_service.resolve("") is None

    def test_expired_token_returns_none_and_row_deleted(self, seeded_service):
        issued = seeded_service.login("academic", "academic12345")
        token = issued["token"]
        with seeded_service.db.session_scope() as session:
            row = session.query(ConsoleSession).filter(ConsoleSession.token == token).one()
            row.expires_at = datetime.now() - timedelta(minutes=1)

        assert seeded_service.resolve(token) is None
        with seeded_service.db.session_scope() as session:
            assert session.query(ConsoleSession).filter(
                ConsoleSession.token == token
            ).count() == 0

    def test_logout_then_resolve_none(self, seeded_service):
        issued = seeded_service.login("support", "support12345")
        token = issued["token"]
        assert seeded_service.resolve(token) is not None

        seeded_service.logout(token)
        assert seeded_service.resolve(token) is None

    def test_logout_is_idempotent(self, seeded_service):
        issued = seeded_service.login("support", "support12345")
        seeded_service.logout(issued["token"])
        seeded_service.logout(issued["token"])
        seeded_service.logout(None)
        seeded_service.logout("")


# ----------------------------------------------------------------------
# 5. 依赖矩阵(本 spec 核心验收)
# ----------------------------------------------------------------------

_PROBE_MOUNTED = False


def _mount_probe_router(app) -> None:
    """把受保护探针端点挂到既有 app 上(不触碰 web/routes.py)。"""
    from fastapi import APIRouter, Depends
    from api.core.console_auth import require_console_user

    probe = APIRouter(prefix="/__probe/console", tags=["probe"])

    @probe.get("/any")
    async def _any(user: dict = Depends(require_console_user())):
        return {"username": user["username"], "role": user["role"]}

    @probe.get("/admin-only")
    async def _admin_only(user: dict = Depends(require_console_user("admin"))):
        return {"username": user["username"]}

    @probe.get("/support-only")
    async def _support_only(user: dict = Depends(require_console_user("support"))):
        return {"username": user["username"]}

    app.include_router(probe)

class TestDependencyMatrix:

    @pytest.fixture
    def guarded_client(self, client):
        """在 app 上就地挂受保护测试端点,不触碰 web/routes.py。

        app 是模块级单例,重复 include 会产生重复 operation id 警告并污染
        后续 openapi 用例,所以用一次性守卫只挂一次。
        """
        global _PROBE_MOUNTED
        if not _PROBE_MOUNTED:
            _mount_probe_router(client.app)
            _PROBE_MOUNTED = True
        return client

    def test_no_cookie_is_401(self, guarded_client):
        assert guarded_client.get("/__probe/console/any").status_code == 401

    def test_forged_cookie_is_401(self, guarded_client):
        guarded_client.cookies.set(CONSOLE_COOKIE_NAME, "forged-token")
        assert guarded_client.get("/__probe/console/any").status_code == 401

    def test_matching_role_is_200(self, guarded_client):
        r = guarded_client.post("/api/console/login",
                                json={"username": "admin", "password": "admin12345"})
        assert r.status_code == 200, r.text
        assert guarded_client.get("/__probe/console/admin-only").status_code == 200

    def test_mismatched_role_is_403(self, guarded_client):
        r = guarded_client.post("/api/console/login",
                                json={"username": "support", "password": "support12345"})
        assert r.status_code == 200, r.text
        assert guarded_client.get("/__probe/console/admin-only").status_code == 403
        assert guarded_client.get("/__probe/console/support-only").status_code == 200

    @pytest.mark.parametrize("username,password", [
        ("admin", "admin12345"),
        ("support", "support12345"),
        ("academic", "academic12345"),
    ])
    def test_any_logged_in_role_passes_roleless_dependency(
        self, guarded_client, username, password
    ):
        r = guarded_client.post("/api/console/login",
                                json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        probe = guarded_client.get("/__probe/console/any")
        assert probe.status_code == 200
        assert probe.json()["username"] == username


# ----------------------------------------------------------------------
# 6. 端到端 login → me → logout → me
# ----------------------------------------------------------------------

class TestConsoleAuthEndpoints:

    def test_login_sets_cookie_and_me_hides_hash(self, client):
        r = client.post("/api/console/login",
                        json={"username": "admin", "password": "admin12345"})
        assert r.status_code == 200, r.text
        assert CONSOLE_COOKIE_NAME in r.cookies
        body = r.json()["data"]
        assert body["username"] == "admin"
        assert "password_hash" not in body

        me = client.get("/api/console/me")
        assert me.status_code == 200, me.text
        me_data = me.json()["data"]
        assert set(me_data) == {"username", "role", "display_name"}
        assert "password_hash" not in me_data
        assert "token" not in me_data

    def test_logout_invalidates_session(self, client):
        r = client.post("/api/console/login",
                        json={"username": "support", "password": "support12345"})
        assert r.status_code == 200, r.text
        assert client.get("/api/console/me").status_code == 200

        out = client.post("/api/console/logout")
        assert out.status_code == 200, out.text
        assert client.get("/api/console/me").status_code == 401

    def test_logout_without_login_is_200(self, client):
        assert client.post("/api/console/logout").status_code == 200
        assert client.get("/api/console/me").status_code == 401

    def test_login_failure_is_401_without_leaking(self, client):
        r = client.post("/api/console/login",
                        json={"username": "ghost", "password": "whatever"})
        assert r.status_code == 401
        assert CONSOLE_COOKIE_NAME not in r.cookies

    def test_me_without_login_is_401(self, client):
        assert client.get("/api/console/me").status_code == 401
