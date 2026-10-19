"""老师面验收测试（实施 C：老师端最小闭环）

覆盖：
1. 面隔离矩阵：未登录 401 / teacher 进 /teacher 200 / teacher 进 /console 403 / 其它角色进 /teacher 403
2. ensure_default_teacher_accounts 幂等（逐 username 查缺补漏）
3. teacher 身份：/api/teacher/me 返回会话绑定的 teacher_id
4. 只含本老师的课
5. 越权写操作：他人课 accept / reject → 404，且状态不变
6. accept / reject 状态机 + 档期冲突复检
7. reject 不触发退费（证据：cancel_lesson / CancelHandler.cancel_lesson 被替换为抛错，且退费痕迹字段不变）
8. 红线：请求里的 teacher_id（query / body）被忽略
9. 未登录调 /api/teacher/* → 401

约定：不用 pytest-asyncio；走路由用 TestClient；直连服务显式传 tmp_path 文件库；不硬编码日期。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.console_auth import CONSOLE_COOKIE_NAME

TEACHER_NAMES = ["张明", "李静", "王强", "陈雪", "刘洋", "周敏", "高远", "苏雨"]
TEACHER_PASSWORD = "teacher12345"


def _db_url(tmp_path, name: str) -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


def _at(days: int, hour: int = 10, minute: int = 0) -> datetime:
    """相对今天的时刻（不硬编码日期）。"""
    base = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)


def _seed_teachers(url: str, count: int = len(TEACHER_NAMES)) -> None:
    from db.base.session_manager import SessionManager
    from db.models import Teacher

    sm = SessionManager(url)
    try:
        with sm.session_scope() as session:
            for idx, name in enumerate(TEACHER_NAMES[:count], start=1):
                session.add(Teacher(id=idx, name=name, is_active=True))
    finally:
        sm.close()


def _seed_student(url: str, name: str) -> int:
    from db.base.session_manager import SessionManager
    from db.models import StudentProfile

    sm = SessionManager(url)
    try:
        with sm.session_scope() as session:
            profile = StudentProfile(name=name)
            session.add(profile)
            session.flush()
            return profile.id
    finally:
        sm.close()


@pytest.fixture
def face_url(tmp_path) -> str:
    return _db_url(tmp_path, "teacher_face.db")


@pytest.fixture
def face_env(monkeypatch, face_url):
    """TestClient + 控制台账号 / 课时服务全部指向同一临时文件库。"""
    from fastapi.testclient import TestClient
    from api.core import console_auth as console_dep
    from app import app
    from db.db_router import DatabaseRouter
    from services import lesson_service as lesson_module
    from services.console_auth_service import ConsoleAuthService

    original_console_init = ConsoleAuthService.__init__

    def _patched_console_init(self, db_path=None, *args, **kwargs):
        original_console_init(self, db_path=db_path or face_url)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_console_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)
    # LessonService() 默认构造 DatabaseRouter()（生产库）；这里换成临时库
    monkeypatch.setattr(lesson_module, "DatabaseRouter", lambda *a, **k: DatabaseRouter(face_url))

    _seed_teachers(face_url)
    ConsoleAuthService(db_path=face_url).ensure_default_users()
    ConsoleAuthService(db_path=face_url).ensure_default_teacher_accounts()

    return TestClient(app)


def _login(client, username: str, password: str):
    client.cookies.clear()
    response = client.post(
        "/api/console/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    assert CONSOLE_COOKIE_NAME in response.cookies
    return response


def _login_teacher(client, teacher_id: int):
    return _login(client, f"teacher{teacher_id}", TEACHER_PASSWORD)


def _lessons_repo(url: str):
    from db.db_router import DatabaseRouter

    return DatabaseRouter(url).lessons


def _add_lesson(url: str, **kwargs) -> int:
    defaults = dict(
        lesson_type="trial", subject="数学", duration_minutes=60,
        status="pending", series_id=None, series_index=None,
        package_id=None, price=None,
    )
    defaults.update(kwargs)
    return _lessons_repo(url).add_lesson(**defaults)


def _status_of(url: str, lesson_id: int) -> str:
    return _lessons_repo(url).get_lesson_by_id(lesson_id)["status"]


@pytest.fixture
def two_teachers(face_url):
    """老师 A=1、B=2 各造若干 pending 课（不同时段，互不冲突）。"""
    student_a = _seed_student(face_url, "学生甲")
    student_b = _seed_student(face_url, "学生乙")
    a_ids = [
        _add_lesson(face_url, teacher_id=1, student_id=student_a, scheduled_at=_at(2)),
        _add_lesson(face_url, teacher_id=1, student_id=student_a, scheduled_at=_at(3, 15)),
    ]
    b_ids = [
        _add_lesson(face_url, teacher_id=2, student_id=student_b, scheduled_at=_at(2)),
        _add_lesson(face_url, teacher_id=2, student_id=student_b, scheduled_at=_at(3, 15)),
    ]
    return {"a": a_ids, "b": b_ids, "student_a": student_a, "student_b": student_b}


# ----------------------------------------------------------------------
# 1. 面隔离矩阵
# ----------------------------------------------------------------------

class TestSurfaceMatrix:

    def test_teacher_home_without_cookie_is_401(self, face_env):
        face_env.cookies.clear()
        assert face_env.get("/teacher").status_code == 401

    def test_teacher_login_page_is_public_and_has_no_credentials(self, face_env):
        face_env.cookies.clear()
        r = face_env.get("/teacher/login")
        assert r.status_code == 200, r.text
        assert "/api/console/login" in r.text
        assert TEACHER_PASSWORD not in r.text

    def test_teacher_home_is_200_after_teacher_login(self, face_env):
        _login_teacher(face_env, 1)
        r = face_env.get("/teacher")
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers.get("content-type", "")

    @pytest.mark.parametrize("path", ["/console", "/console/knowledge", "/console/human-handoff"])
    def test_teacher_cannot_enter_console_pages(self, face_env, path):
        _login_teacher(face_env, 1)
        assert face_env.get(path).status_code == 403

    @pytest.mark.parametrize("path", ["/api/knowledge/", "/api/handoffs?status=pending"])
    def test_teacher_cannot_enter_console_apis(self, face_env, path):
        _login_teacher(face_env, 1)
        assert face_env.get(path).status_code == 403

    @pytest.mark.parametrize("username", ["admin", "support", "academic"])
    def test_staff_roles_cannot_enter_teacher_home(self, face_env, username):
        _login(face_env, username, f"{username}12345")
        assert face_env.get("/teacher").status_code == 403

    def test_admin_console_home_still_200(self, face_env):
        """收紧首页后的回归：admin 仍能进 /console。"""
        _login(face_env, "admin", "admin12345")
        assert face_env.get("/console").status_code == 200


# ----------------------------------------------------------------------
# 2. ensure_default_teacher_accounts 幂等（显式临时库，不 patch）
# ----------------------------------------------------------------------

class TestEnsureDefaultTeacherAccounts:

    def test_idempotent_and_restores_only_missing(self, tmp_path):
        from db.models import ConsoleUser
        from services.console_auth_service import ConsoleAuthService

        url = _db_url(tmp_path, "accounts.db")
        _seed_teachers(url)
        svc = ConsoleAuthService(db_path=url)
        try:
            assert svc.ensure_default_teacher_accounts() == len(TEACHER_NAMES)
            assert svc.ensure_default_teacher_accounts() == 0

            with svc.db.session_scope() as session:
                session.query(ConsoleUser).filter(ConsoleUser.username == "teacher3").delete()

            assert svc.ensure_default_teacher_accounts() == 1
        finally:
            svc.close()

    def test_teacher_login_binds_teacher_id_and_staff_is_none(self, tmp_path):
        from services.console_auth_service import ConsoleAuthService

        url = _db_url(tmp_path, "bind.db")
        _seed_teachers(url)
        svc = ConsoleAuthService(db_path=url)
        try:
            svc.ensure_default_users()
            svc.ensure_default_teacher_accounts()
            teacher = svc.login("teacher2", TEACHER_PASSWORD)
            staff = svc.login("admin", "admin12345")
            assert teacher["role"] == "teacher"
            assert teacher["teacher_id"] == 2
            assert staff["teacher_id"] is None
            assert svc.resolve(teacher["token"])["teacher_id"] == 2
        finally:
            svc.close()


# ----------------------------------------------------------------------
# 3. 老师身份
# ----------------------------------------------------------------------

class TestTeacherIdentity:

    def test_me_returns_bound_teacher(self, face_env):
        _login_teacher(face_env, 1)
        r = face_env.get("/api/teacher/me")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"teacher_id": 1, "name": "张明", "username": "teacher1"}

    def test_me_for_staff_is_403(self, face_env):
        _login(face_env, "admin", "admin12345")
        assert face_env.get("/api/teacher/me").status_code == 403


# ----------------------------------------------------------------------
# 4. 只含本老师的课
# ----------------------------------------------------------------------

class TestLessonListScope:

    def test_list_contains_only_own_lessons(self, face_env, two_teachers):
        _login_teacher(face_env, 1)
        r = face_env.get("/api/teacher/lessons?status=pending")
        assert r.status_code == 200, r.text
        ids = {row["id"] for row in r.json()}
        assert ids == set(two_teachers["a"])
        assert not ids & set(two_teachers["b"])

    def test_default_status_is_pending(self, face_env, two_teachers):
        _login_teacher(face_env, 1)
        ids = {row["id"] for row in face_env.get("/api/teacher/lessons").json()}
        assert ids == set(two_teachers["a"])

    def test_student_name_is_included(self, face_env, two_teachers):
        _login_teacher(face_env, 1)
        rows = face_env.get("/api/teacher/lessons").json()
        assert rows and all(row["student_name"] == "学生甲" for row in rows)

    def test_status_filter_returns_matching_only(self, face_env, face_url, two_teachers):
        confirmed_id = _add_lesson(
            face_url, teacher_id=1, student_id=two_teachers["student_a"],
            scheduled_at=_at(5), status="confirmed",
        )
        _login_teacher(face_env, 1)
        rows = face_env.get("/api/teacher/lessons?status=confirmed").json()
        assert [row["id"] for row in rows] == [confirmed_id]

    @pytest.mark.parametrize("bad", ["bogus", "rescheduled", ""])
    def test_invalid_status_is_422(self, face_env, bad):
        _login_teacher(face_env, 1)
        assert face_env.get(f"/api/teacher/lessons?status={bad}").status_code == 422


# ----------------------------------------------------------------------
# 5. 越权写操作：他人课一律 404，且状态不变
# ----------------------------------------------------------------------

class TestCrossTeacherWrites:

    def test_accept_others_lesson_is_404_and_status_unchanged(self, face_env, face_url, two_teachers):
        target = two_teachers["b"][0]
        before = _status_of(face_url, target)
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/accept")
        assert r.status_code == 404
        assert _status_of(face_url, target) == before == "pending"

    def test_reject_others_lesson_is_404_and_status_unchanged(self, face_env, face_url, two_teachers):
        target = two_teachers["b"][0]
        before = _status_of(face_url, target)
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/reject", json={"reason": "越权试探"})
        assert r.status_code == 404
        assert _status_of(face_url, target) == before == "pending"

    def test_nonexistent_lesson_is_404_same_as_foreign(self, face_env):
        _login_teacher(face_env, 1)
        assert face_env.post("/api/teacher/lessons/999999/accept").status_code == 404


# ----------------------------------------------------------------------
# 6. accept / reject 状态机 + 档期复检
# ----------------------------------------------------------------------

class TestAcceptReject:

    def test_accept_pending_becomes_confirmed(self, face_env, face_url, two_teachers):
        target = two_teachers["a"][0]
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/accept")
        assert r.status_code == 200, r.text
        assert r.json() == {"lesson_id": target, "status": "confirmed"}
        assert _status_of(face_url, target) == "confirmed"

    @pytest.mark.parametrize("state", ["confirmed", "completed", "cancelled"])
    def test_accept_non_pending_is_409_and_status_unchanged(self, face_env, face_url, two_teachers, state):
        target = two_teachers["a"][0]
        _lessons_repo(face_url).update_lesson(target, status=state)
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/accept")
        assert r.status_code == 409, r.text
        assert _status_of(face_url, target) == state

    def test_accept_conflicting_slot_is_409_and_stays_pending(self, face_env, face_url, two_teachers):
        """老师已有一节确认课与待确认课重叠 → 确认失败，状态仍 pending。"""
        _add_lesson(
            face_url, teacher_id=1, student_id=two_teachers["student_a"],
            scheduled_at=_at(6, 10), status="confirmed", duration_minutes=60,
        )
        pending_overlap = _add_lesson(
            face_url, teacher_id=1, student_id=two_teachers["student_a"],
            scheduled_at=_at(6, 10) + timedelta(minutes=30), status="pending", duration_minutes=60,
        )
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{pending_overlap}/accept")
        assert r.status_code == 409, r.text
        assert _status_of(face_url, pending_overlap) == "pending"

    def test_reject_pending_becomes_cancelled_and_echoes_reason(self, face_env, face_url, two_teachers):
        target = two_teachers["a"][0]
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/reject", json={"reason": "时间冲突"})
        assert r.status_code == 200, r.text
        assert r.json() == {"lesson_id": target, "status": "cancelled", "reason": "时间冲突"}
        assert _status_of(face_url, target) == "cancelled"

    def test_reject_with_empty_reason_is_allowed(self, face_env, face_url, two_teachers):
        target = two_teachers["a"][0]
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/reject", json={"reason": None})
        assert r.status_code == 200, r.text
        assert _status_of(face_url, target) == "cancelled"

    @pytest.mark.parametrize("state", ["confirmed", "completed", "cancelled"])
    def test_reject_non_pending_is_409_and_status_unchanged(self, face_env, face_url, two_teachers, state):
        target = two_teachers["a"][0]
        _lessons_repo(face_url).update_lesson(target, status=state)
        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/reject", json={"reason": "x"})
        assert r.status_code == 409, r.text
        assert _status_of(face_url, target) == state


# ----------------------------------------------------------------------
# 7. reject 不触发退费
#
# 退费证据（grep 结果）：agents/lesson/cancel_handler.py 的 CancelHandler.cancel_lesson
# 才会调用 calculate_refund，并写入 parent_feedback（"[退课原因]"）与系列 cancelled_lessons +1；
# calculate_refund 本身只返回 refund_info 给 /api/lessons/{id}/cancel 的响应，未落库。
# 因此断言：CancelHandler.cancel_lesson 与 LessonService.cancel_lesson 均被替换为抛错（调用即失败），
# 且退费痕迹字段（parent_feedback、系列 cancelled_lessons）保持不变。
# ----------------------------------------------------------------------

class TestRejectDoesNotRefund:

    def test_reject_does_not_touch_refund_side_effects(self, monkeypatch, face_env, face_url, two_teachers):
        from agents.lesson.cancel_handler import CancelHandler
        from services.lesson_service import LessonService
        from datetime import date

        def _forbidden(*args, **kwargs):
            raise AssertionError("拒绝路径不得调用退费语义的 cancel_lesson")

        monkeypatch.setattr(CancelHandler, "cancel_lesson", _forbidden)
        monkeypatch.setattr(LessonService, "cancel_lesson", _forbidden)

        repo = _lessons_repo(face_url)
        series_id = repo.add_series(
            teacher_id=1, student_id=two_teachers["student_a"],
            start_date=date.today(), end_date=date.today() + timedelta(days=7),
            weekly_pattern=[], total_lessons=1,
        )
        target = _add_lesson(
            face_url, teacher_id=1, student_id=two_teachers["student_a"],
            scheduled_at=_at(7), series_id=series_id, series_index=1,
        )
        lessons_before = len(repo.get_lessons_by_teacher(1))

        _login_teacher(face_env, 1)
        r = face_env.post(f"/api/teacher/lessons/{target}/reject", json={"reason": "不退费验证"})
        assert r.status_code == 200, r.text

        lesson = repo.get_lesson_by_id(target)
        assert lesson["status"] == "cancelled"
        assert lesson["parent_feedback"] is None           # cancel_handler 会写入 "[退课原因]"
        assert len(repo.get_lessons_by_teacher(1)) == lessons_before
        series = repo.get_series_by_id(series_id)
        assert series["cancelled_lessons"] == 0           # cancel_handler 会 +1
        assert series["completed_lessons"] == 0


# ----------------------------------------------------------------------
# 8. 红线：teacher_id 只来自会话
# ----------------------------------------------------------------------

class TestTeacherIdRedLine:

    def test_query_teacher_id_is_ignored_on_list(self, face_env, two_teachers):
        _login_teacher(face_env, 1)
        r = face_env.get("/api/teacher/lessons?status=pending&teacher_id=2")
        assert r.status_code == 200, r.text
        ids = {row["id"] for row in r.json()}
        assert ids == set(two_teachers["a"])

    def test_query_teacher_id_is_ignored_on_me(self, face_env):
        _login_teacher(face_env, 1)
        assert face_env.get("/api/teacher/me?teacher_id=2").json()["teacher_id"] == 1

    def test_body_teacher_id_does_not_redirect_reject(self, face_env, face_url, two_teachers):
        own = two_teachers["a"][0]
        foreign = two_teachers["b"][0]
        _login_teacher(face_env, 1)
        r = face_env.post(
            f"/api/teacher/lessons/{own}/reject",
            json={"reason": "x", "teacher_id": 2},
        )
        assert r.status_code == 200, r.text
        assert _status_of(face_url, own) == "cancelled"
        assert _status_of(face_url, foreign) == "pending"

    def test_self_claimed_teacher_id_does_not_grant_foreign_accept(self, face_env, face_url, two_teachers):
        foreign = two_teachers["b"][0]
        _login_teacher(face_env, 1)
        r = face_env.post(
            f"/api/teacher/lessons/{foreign}/accept?teacher_id=1",
            json={"teacher_id": 1},
        )
        assert r.status_code == 404, r.text
        assert _status_of(face_url, foreign) == "pending"


# ----------------------------------------------------------------------
# 9. 未登录 / 非老师调用 /api/teacher/* → 401 / 403
# ----------------------------------------------------------------------

class TestUnauthenticatedApi:

    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/teacher/me"),
        ("GET", "/api/teacher/lessons?status=pending"),
        ("POST", "/api/teacher/lessons/1/accept"),
        ("POST", "/api/teacher/lessons/1/reject"),
    ])
    def test_without_cookie_is_401(self, face_env, method, path):
        face_env.cookies.clear()
        r = face_env.request(method, path, json={"reason": "x"})
        assert r.status_code == 401, r.text


# ----------------------------------------------------------------------
# 10. 老师端「待确认调课」区块接线（页面级）
# ----------------------------------------------------------------------

class TestTeacherChangeRequestBlock:

    def test_teacher_home_wires_change_request_block_and_requires_login(self, face_env):
        """老师页 HTML 含「待确认调课」区块与其接口路径；未登录仍 401（回归）。"""
        _login_teacher(face_env, 1)
        r = face_env.get("/teacher")
        assert r.status_code == 200, r.text
        assert "待确认调课" in r.text
        assert "/api/teacher/change-requests" in r.text

        face_env.cookies.clear()
        assert face_env.get("/teacher").status_code == 401
