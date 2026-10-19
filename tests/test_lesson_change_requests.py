"""调课需老师确认（C2a）测试

覆盖：
1. 创建：落 pending、old_scheduled_at 快照、课时时间不动
2. 幂等：同一课时再发一次 → 同一 request.id + created=False，仍只有 1 条 pending
3. 守卫：课时不存在 404；completed/cancelled 409；老师/学生冲突 409 且不落库
4. 越权：老师 A 对老师 B 的请求 approve/reject → 404，请求仍 pending，课时未变
5. approve 正常：approved + decided_*，课时真的改期
6. approve 级联：this_and_following 改目标及之后的课时
7. approve 时才发现冲突：409，请求仍 pending，课时未变
8. reject：rejected + reason，课时不动；已 rejected 再 approve → 409
9. 面隔离：未登录 401；admin 403

约定：不用 pytest-asyncio；走路由用 TestClient；直连服务显式传 tmp_path 文件库；
不硬编码日期（datetime.now 相对计算）。
路由测试把 services.lesson_change_service.DEFAULT_DB_PATH 与 RescheduleHandler 内部的
db.db_router.DatabaseRouter 一并指向临时库 —— approve 会真改课时，绝不能落到生产库。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.console_auth import CONSOLE_COOKIE_NAME
from db import db_router as db_router_module
from db.db_router import DatabaseRouter
from db.base.session_manager import SessionManager
from db.models import Teacher, StudentProfile
from services.lesson_change_service import LessonChangeError, LessonChangeService

TEACHER_NAMES = ["张明", "李静", "王强", "陈雷", "刘磊", "周敏", "高远", "苏雦"]
TEACHER_PASSWORD = "teacher12345"


def _db_url(tmp_path, name: str) -> str:
    return f"sqlite:///{(tmp_path / name).as_posix()}"


def _at(days: int, hour: int = 10, minute: int = 0) -> datetime:
    """相对今天的时刻（不硬编码日期）。"""
    base = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)


# 全周全天开放:让测试只关心冲突/级联,不被默认档期（availability）干扰
OPEN_AVAILABILITY = [
    {"day": day, "slots": ["00:00-23:59"]}
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
]


def _seed_teachers(url: str) -> None:
    sm = SessionManager(url)
    try:
        with sm.session_scope() as session:
            for idx, name in enumerate(TEACHER_NAMES, start=1):
                session.add(Teacher(id=idx, name=name, is_active=True,
                                    availability=OPEN_AVAILABILITY))
    finally:
        sm.close()


def _seed_student(url: str, name: str) -> int:
    sm = SessionManager(url)
    try:
        with sm.session_scope() as session:
            profile = StudentProfile(name=name)
            session.add(profile)
            session.flush()
            return profile.id
    finally:
        sm.close()


def _add_lesson(url: str, **kwargs) -> int:
    defaults = dict(
        lesson_type="regular", subject="数学", duration_minutes=60,
        status="confirmed", series_id=None, series_index=None,
        package_id=None, price=None,
    )
    defaults.update(kwargs)
    return DatabaseRouter(url).lessons.add_lesson(**defaults)


def _lesson(url: str, lesson_id: int) -> dict:
    return DatabaseRouter(url).lessons.get_lesson_by_id(lesson_id)


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


@pytest.fixture
def cr_url(tmp_path) -> str:
    return _db_url(tmp_path, "lesson_change.db")


@pytest.fixture
def base(cr_url):
    """老师 1..8 + 两个学生(A/B)。"""
    _seed_teachers(cr_url)
    return {
        "url": cr_url,
        "student_a": _seed_student(cr_url, "学生甲"),
        "student_b": _seed_student(cr_url, "学生乙"),
    }


@pytest.fixture
def svc(cr_url, base):
    service = LessonChangeService(db_path=cr_url)
    yield service
    service.close()


@pytest.fixture
def cr_env(monkeypatch, base):
    """TestClient + 控制台账号 + 调课服务/RescheduleHandler 全部指向临时库。"""
    from fastapi.testclient import TestClient
    from api.core import console_auth as console_dep
    from app import app
    from services import lesson_change_service as change_module
    from services.console_auth_service import ConsoleAuthService

    url = base["url"]
    original_console_init = ConsoleAuthService.__init__

    def _patched_console_init(self, db_path=None, *args, **kwargs):
        original_console_init(self, db_path=db_path or url)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_console_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)
    # 课时请求服务的默认库 → 临时库
    monkeypatch.setattr(change_module, "DEFAULT_DB_PATH", url)
    # RescheduleHandler / ConflictDetector 内部直接 DatabaseRouter() → 临时库（否则 approve 会碰生产库）
    real_router = db_router_module.DatabaseRouter
    monkeypatch.setattr(db_router_module, "DatabaseRouter", lambda *a, **k: real_router(url))

    ConsoleAuthService(db_path=url).ensure_default_users()
    ConsoleAuthService(db_path=url).ensure_default_teacher_accounts()
    return TestClient(app)


# ----------------------------------------------------------------------
# 1-2. 创建 & 幂等（服务直连，显式 tmp_path）
# ----------------------------------------------------------------------

class TestCreateRequest:

    def test_create_inserts_pending_and_keeps_lesson_time(self, base, svc):
        url = base["url"]
        original = _at(2)
        lid = _add_lesson(url, teacher_id=1, student_id=base["student_a"], scheduled_at=original)

        result = svc.create_request(lid, _at(3, 15))

        assert result["created"] is True
        req = result["request"]
        assert req["status"] == "pending"
        assert req["old_scheduled_at"] == original
        assert req["new_scheduled_at"] == _at(3, 15)
        rows = svc.list_for_lesson(lid)
        assert len(rows) == 1
        assert _lesson(url, lid)["scheduled_at"] == original

    def test_repeat_create_is_idempotent(self, base, svc):
        url = base["url"]
        lid = _add_lesson(url, teacher_id=1, student_id=base["student_a"], scheduled_at=_at(2))

        first = svc.create_request(lid, _at(3, 15))
        second = svc.create_request(lid, _at(4, 15))

        assert second["created"] is False
        assert second["request"]["id"] == first["request"]["id"]
        pending = [r for r in svc.list_for_lesson(lid) if r["status"] == "pending"]
        assert len(pending) == 1


# ----------------------------------------------------------------------
# 3. 守卫
# ----------------------------------------------------------------------

class TestCreateGuards:

    def test_missing_lesson_is_404(self, svc):
        with pytest.raises(LessonChangeError) as exc:
            svc.create_request(999999, _at(3))
        assert exc.value.status_code == 404

    @pytest.mark.parametrize("status", ["completed", "cancelled"])
    def test_closed_lesson_is_409(self, base, svc, status):
        lid = _add_lesson(base["url"], teacher_id=1, student_id=base["student_a"],
                          scheduled_at=_at(-2), status=status)
        with pytest.raises(LessonChangeError) as exc:
            svc.create_request(lid, _at(3))
        assert exc.value.status_code == 409
        assert svc.list_for_lesson(lid) == []

    def test_teacher_conflict_is_409_and_not_persisted(self, base, svc):
        url = base["url"]
        lid = _add_lesson(url, teacher_id=1, student_id=base["student_a"], scheduled_at=_at(2))
        _add_lesson(url, teacher_id=1, student_id=base["student_b"], scheduled_at=_at(5, 10))

        with pytest.raises(LessonChangeError) as exc:
            svc.create_request(lid, _at(5, 10))
        assert exc.value.status_code == 409
        assert svc.list_for_lesson(lid) == []

    def test_student_conflict_is_409_and_not_persisted(self, base, svc):
        url = base["url"]
        lid = _add_lesson(url, teacher_id=1, student_id=base["student_a"], scheduled_at=_at(2))
        # 同一学生在别的老师那里已有课，新时间与之冲突
        _add_lesson(url, teacher_id=2, student_id=base["student_a"], scheduled_at=_at(6, 10))

        with pytest.raises(LessonChangeError) as exc:
            svc.create_request(lid, _at(6, 10))
        assert exc.value.status_code == 409
        assert svc.list_for_lesson(lid) == []


# ----------------------------------------------------------------------
# 4-8. 老师决策（走路由）
# ----------------------------------------------------------------------

def _pending_request(base, svc, teacher_id: int, new_time: datetime, mode: str = "only_this",
                     lesson_id: int | None = None) -> tuple[int, int]:
    """为 teacher_id 的课造一条 pending 请求，返回 (lesson_id, request_id)。"""
    url = base["url"]
    if lesson_id is None:
        lesson_id = _add_lesson(url, teacher_id=teacher_id, student_id=base["student_a"],
                                scheduled_at=_at(2))
    result = svc.create_request(lesson_id, new_time, mode=mode)
    return lesson_id, result["request"]["id"]


class TestCrossTeacher:

    def test_other_teacher_approve_and_reject_are_404(self, cr_env, base, svc):
        url = base["url"]
        lid, rid = _pending_request(base, svc, teacher_id=2, new_time=_at(3, 15))
        before = _lesson(url, lid)["scheduled_at"]

        _login_teacher(cr_env, 1)  # 老师 A 操作老师 B 的请求
        assert cr_env.post(f"/api/teacher/change-requests/{rid}/approve").status_code == 404
        assert cr_env.post(f"/api/teacher/change-requests/{rid}/reject",
                           json={"reason": "x"}).status_code == 404

        still = svc.list_for_lesson(lid)[0]
        assert still["status"] == "pending"
        assert _lesson(url, lid)["scheduled_at"] == before

    def test_teacher_list_only_contains_own_requests(self, cr_env, base, svc):
        _, mine = _pending_request(base, svc, teacher_id=1, new_time=_at(3, 15))
        _, theirs = _pending_request(base, svc, teacher_id=2, new_time=_at(3, 15))

        _login_teacher(cr_env, 1)
        r = cr_env.get("/api/teacher/change-requests?status=pending")
        assert r.status_code == 200, r.text
        ids = [item["id"] for item in r.json()]
        assert mine in ids and theirs not in ids


class TestApprove:

    def test_approve_applies_new_time(self, cr_env, base, svc):
        url = base["url"]
        new_time = _at(3, 15)
        lid, rid = _pending_request(base, svc, teacher_id=1, new_time=new_time)

        _login_teacher(cr_env, 1)
        r = cr_env.post(f"/api/teacher/change-requests/{rid}/approve")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "approved"
        assert body["decided_by_teacher_id"] == 1
        assert body["decided_at"] is not None

        lesson = _lesson(url, lid)
        assert lesson["scheduled_at"] == new_time

    def test_approve_cascades_this_and_following(self, cr_env, base, svc):
        url = base["url"]
        series_id = DatabaseRouter(url).lessons.add_series(
            teacher_id=1, student_id=base["student_a"],
            start_date=_at(2).date(), end_date=_at(16).date(),
            weekly_pattern=[], total_lessons=3,
        )
        lids = [
            _add_lesson(url, teacher_id=1, student_id=base["student_a"],
                        scheduled_at=_at(2 + 7 * i), series_id=series_id, series_index=i)
            for i in range(3)
        ]
        before = [_lesson(url, lid)["scheduled_at"] for lid in lids]

        shift = timedelta(hours=8)
        _, rid = _pending_request(base, svc, teacher_id=1, new_time=before[0] + shift,
                                  mode="this_and_following", lesson_id=lids[0])

        _login_teacher(cr_env, 1)
        r = cr_env.post(f"/api/teacher/change-requests/{rid}/approve")
        assert r.status_code == 200, r.text

        moved = [_lesson(url, lid)["scheduled_at"] for lid in lids]
        assert moved[0] == before[0] + shift
        assert moved[1] == before[1] + shift
        assert moved[2] == before[2] + shift

    def test_approve_conflict_found_late_keeps_pending(self, cr_env, base, svc):
        url = base["url"]
        new_time = _at(4, 10)
        lid, rid = _pending_request(base, svc, teacher_id=1, new_time=new_time)
        before = _lesson(url, lid)["scheduled_at"]

        # 创建之后，老师在新时间又被占用
        _add_lesson(url, teacher_id=1, student_id=base["student_b"], scheduled_at=new_time)

        _login_teacher(cr_env, 1)
        r = cr_env.post(f"/api/teacher/change-requests/{rid}/approve")
        assert r.status_code == 409, r.text

        req = svc.list_for_lesson(lid)[0]
        assert req["status"] == "pending"
        assert _lesson(url, lid)["scheduled_at"] == before


class TestReject:

    def test_reject_stores_reason_and_keeps_time(self, cr_env, base, svc):
        url = base["url"]
        lid, rid = _pending_request(base, svc, teacher_id=1, new_time=_at(3, 15))
        before = _lesson(url, lid)["scheduled_at"]

        _login_teacher(cr_env, 1)
        r = cr_env.post(f"/api/teacher/change-requests/{rid}/reject", json={"reason": "当天有事"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "rejected"

        req = svc.list_for_lesson(lid)[0]
        assert req["status"] == "rejected"
        assert req["reason"] == "当天有事"
        assert req["decided_by_teacher_id"] == 1
        assert req["decided_at"] is not None
        assert _lesson(url, lid)["scheduled_at"] == before

    def test_approve_after_reject_is_409(self, cr_env, base, svc):
        lid, rid = _pending_request(base, svc, teacher_id=1, new_time=_at(3, 15))

        _login_teacher(cr_env, 1)
        assert cr_env.post(f"/api/teacher/change-requests/{rid}/reject", json={}).status_code == 200
        r = cr_env.post(f"/api/teacher/change-requests/{rid}/approve")
        assert r.status_code == 409, r.text
        assert svc.list_for_lesson(lid)[0]["status"] == "rejected"


# ----------------------------------------------------------------------
# 9. 面隔离
# ----------------------------------------------------------------------

class TestSurfaceIsolation:

    def test_unauthenticated_teacher_list_is_401(self, cr_env):
        cr_env.cookies.clear()
        assert cr_env.get("/api/teacher/change-requests?status=pending").status_code == 401

    def test_admin_teacher_list_is_403(self, cr_env):
        _login(cr_env, "admin", "admin12345")
        assert cr_env.get("/api/teacher/change-requests?status=pending").status_code == 403


# ----------------------------------------------------------------------
# 家长侧 API（路由层）
# ----------------------------------------------------------------------

class TestParentApi:

    def test_post_twice_returns_same_request(self, cr_env, base):
        lid = _add_lesson(base["url"], teacher_id=1, student_id=base["student_a"],
                          scheduled_at=_at(2))
        body = {"new_time": _at(3, 15).isoformat()}

        first = cr_env.post(f"/api/lessons/{lid}/change-requests", json=body)
        second = cr_env.post(f"/api/lessons/{lid}/change-requests", json=body)

        assert first.status_code == 200, first.text
        assert first.json()["created"] is True
        assert second.json()["created"] is False
        assert second.json()["request"]["id"] == first.json()["request"]["id"]

    def test_invalid_mode_is_422(self, cr_env, base):
        lid = _add_lesson(base["url"], teacher_id=1, student_id=base["student_a"],
                          scheduled_at=_at(2))
        r = cr_env.post(f"/api/lessons/{lid}/change-requests",
                        json={"new_time": _at(3).isoformat(), "mode": "all"})
        assert r.status_code == 422

    def test_missing_lesson_is_404(self, cr_env):
        r = cr_env.post("/api/lessons/999999/change-requests",
                        json={"new_time": _at(3).isoformat()})
        assert r.status_code == 404

    def test_get_lists_requests_for_lesson(self, cr_env, base):
        lid = _add_lesson(base["url"], teacher_id=1, student_id=base["student_a"],
                          scheduled_at=_at(2))
        cr_env.post(f"/api/lessons/{lid}/change-requests",
                    json={"new_time": _at(3, 15).isoformat()})

        r = cr_env.get(f"/api/lessons/{lid}/change-requests")
        assert r.status_code == 200, r.text
        assert len(r.json()["data"]) == 1
        assert r.json()["data"][0]["status"] == "pending"
