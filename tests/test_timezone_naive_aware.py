"""naive/aware 时间比较回归（C2d）

背景：生产库 lessons.scheduled_at 全是北京时间 naive；业务层（time_config.parse_datetime）产出 aware。
既有测试全用 aware 造库，从未暴露 `can't compare offset-naive and offset-aware datetimes`。

覆盖：
1-2. 生产形态（naive 课时）+ aware 入参：LessonChangeService.create_request / RescheduleHandler.reschedule 不抛异常
3.   check_teacher_conflict / check_student_conflict：aware 入参与同一时刻的 naive 入参结果一致
4.   naive 入参零变化：冲突/不冲突两个方向双向断言（防止为修 bug 把比较改松）
5.   对话侧端到端：真 change_service + naive 临时库 → 「等待老师确认」，课时时间不变
6.   HTTP 端到端：ISO 带偏移的 new_time → 200 pending_teacher_confirm，课时时间不变

约定：不用 pytest-asyncio；走路由用 TestClient；直连服务显式传 tmp_path 文件库；
课时一律用 naive 北京时间造（模拟生产库）；不硬编码日期（datetime.now 相对计算）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agents.lesson.change_processor import LessonChangeMessageBuilder, LessonChangeProcessor
from agents.lesson.reschedule_handler import RescheduleHandler, RescheduleMode
from db import db_router as db_router_module
from db.base.session_manager import SessionManager
from db.db_router import DatabaseRouter
from db.models import StudentProfile, Teacher
from services.lesson_change_service import LessonChangeService

BEIJING_TZ = timezone(timedelta(hours=8))

OPEN_AVAILABILITY = [
    {"day": day, "slots": ["00:00-23:59"]}
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
]


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream) -> str:
    tokens = []
    async for token in stream:
        tokens.append(token)
    return "".join(tokens)


def _naive_at(days: int, hour: int = 10, minute: int = 0) -> datetime:
    """相对今天的北京时间 naive 时刻（生产库形态）。"""
    base = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)


def _aware(naive: datetime) -> datetime:
    """同一墙上时刻的北京时间 aware（业务层 parse_datetime 的形态）。"""
    return naive.replace(tzinfo=BEIJING_TZ)


def _db_url(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'tz_naive_aware.db').as_posix()}"


def _add_lesson(url: str, naive_time: datetime, **kwargs) -> int:
    """用 naive 时间造课时，模拟生产库。"""
    defaults = dict(
        lesson_type="regular", subject="数学", duration_minutes=60,
        status="confirmed", series_id=None, series_index=None,
        package_id=None, price=None,
    )
    defaults.update(kwargs)
    return DatabaseRouter(url).lessons.add_lesson(scheduled_at=naive_time, **defaults)


def _lesson(url: str, lesson_id: int) -> dict:
    return DatabaseRouter(url).lessons.get_lesson_by_id(lesson_id)


def _seed_neighbours(url: str) -> None:
    """老师 1 / 学生 1 各有一节不重叠的课（6 天后）。

    仓库冲突检测只有在查到已有行时才会做 Python 侧时间比较，
    没有邻居课时 naive/aware 混比根本不会执行，测不出缺陷。
    """
    later = _naive_at(6, 10)
    _add_lesson(url, later, teacher_id=1, student_id=2)
    _add_lesson(url, later, teacher_id=2, student_id=1)


@pytest.fixture
def tz_url(tmp_path) -> str:
    url = _db_url(tmp_path)
    sm = SessionManager(url)
    try:
        with sm.session_scope() as session:
            session.add(Teacher(id=1, name="张明", is_active=True, availability=OPEN_AVAILABILITY))
            session.add(StudentProfile(id=1, name="学生甲"))
            session.add(StudentProfile(id=2, name="学生乙"))
    finally:
        sm.close()
    return url


@pytest.fixture
def point_db_to_tmp(monkeypatch, tz_url):
    """LessonChangeService 默认库 + RescheduleHandler 内部 DatabaseRouter 全部指向临时库。"""
    from services import lesson_change_service as change_module

    monkeypatch.setattr(change_module, "DEFAULT_DB_PATH", tz_url)
    real_router = db_router_module.DatabaseRouter
    monkeypatch.setattr(db_router_module, "DatabaseRouter", lambda *a, **k: real_router(tz_url))
    return tz_url


@pytest.fixture
def svc(tz_url):
    service = LessonChangeService(db_path=tz_url)
    yield service
    service.close()


# ----------------------------------------------------------------------
# 1-2. 生产形态：naive 课时 + aware 入参
# ----------------------------------------------------------------------

class TestProductionShape:

    def test_fixture_lesson_is_naive_like_production(self, tz_url):
        lid = _add_lesson(tz_url, _naive_at(2), teacher_id=1, student_id=1)

        stored = _lesson(tz_url, lid)["scheduled_at"]

        assert stored.tzinfo is None
        assert stored == _naive_at(2)

    def test_create_request_with_aware_new_time_does_not_raise(self, tz_url, svc):
        original = _naive_at(2)
        lid = _add_lesson(tz_url, original, teacher_id=1, student_id=1)
        _seed_neighbours(tz_url)

        result = svc.create_request(lid, _aware(_naive_at(3, 15)))

        assert result["created"] is True
        assert result["request"]["status"] == "pending"
        rows = svc.list_for_lesson(lid)
        assert [row["status"] for row in rows] == ["pending"]
        assert _lesson(tz_url, lid)["scheduled_at"] == original

    def test_reschedule_handler_with_aware_new_time_does_not_raise(self, point_db_to_tmp):
        url = point_db_to_tmp
        lid = _add_lesson(url, _naive_at(2), teacher_id=1, student_id=1)
        _seed_neighbours(url)
        new_naive = _naive_at(3, 15)

        success, message = RescheduleHandler().reschedule(
            lesson_id=lid, new_time=_aware(new_naive), mode=RescheduleMode.ONLY_THIS,
        )

        assert success is True, message
        assert _lesson(url, lid)["scheduled_at"] == new_naive


# ----------------------------------------------------------------------
# 3. 仓库比较：aware 与同一时刻的 naive 结果一致
# ----------------------------------------------------------------------

class TestRepositoryConsistency:

    @pytest.mark.parametrize("start_offset, expected", [
        ((4, 10), True),    # 与 10:00-11:00 课完全重合
        ((4, 10, 30), True),  # 部分重叠
        ((4, 12), False),   # 完全不重叠
        ((4, 11), False),   # 首尾相接不算冲突
    ])
    def test_teacher_conflict_aware_equals_naive(self, tz_url, start_offset, expected):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=1, student_id=1)
        repo = DatabaseRouter(tz_url).lessons
        start_naive = _naive_at(*start_offset)
        end_naive = start_naive + timedelta(minutes=60)

        naive_result = repo.check_teacher_conflict(1, start_naive, end_naive)
        aware_result = repo.check_teacher_conflict(
            1, _aware(start_naive), _aware(end_naive),
        )

        assert aware_result is naive_result
        assert naive_result is expected

    @pytest.mark.parametrize("start_offset, expected", [
        ((4, 10), True),
        ((4, 10, 30), True),
        ((4, 12), False),
        ((4, 11), False),
    ])
    def test_student_conflict_aware_equals_naive(self, tz_url, start_offset, expected):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=1, student_id=1)
        repo = DatabaseRouter(tz_url).lessons
        start_naive = _naive_at(*start_offset)
        end_naive = start_naive + timedelta(minutes=60)

        naive_result = repo.check_student_conflict(1, start_naive, end_naive)
        aware_result = repo.check_student_conflict(
            1, _aware(start_naive), _aware(end_naive),
        )

        assert aware_result is naive_result
        assert naive_result is expected

    def test_get_lessons_by_teacher_accepts_aware_range(self, tz_url):
        inside = _naive_at(4, 10)
        outside = _naive_at(9, 10)
        _add_lesson(tz_url, inside, teacher_id=1, student_id=1)
        _add_lesson(tz_url, outside, teacher_id=1, student_id=2)
        repo = DatabaseRouter(tz_url).lessons

        rows = repo.get_lessons_by_teacher(
            1,
            start_date=_aware(_naive_at(3)),
            end_date=_aware(_naive_at(5)),
        )

        assert [row["scheduled_at"] for row in rows] == [inside]


# ----------------------------------------------------------------------
# 4. naive 入参零变化：双向断言
# ----------------------------------------------------------------------

class TestNaiveInputUnchanged:

    def test_teacher_busy_naive_is_true(self, tz_url):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=1, student_id=1)
        repo = DatabaseRouter(tz_url).lessons

        busy_start = _naive_at(4, 10, 30)
        assert repo.check_teacher_conflict(1, busy_start, busy_start + timedelta(minutes=60)) is True

    def test_teacher_free_naive_is_false(self, tz_url):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=1, student_id=1)
        repo = DatabaseRouter(tz_url).lessons

        free_start = _naive_at(4, 12)
        assert repo.check_teacher_conflict(1, free_start, free_start + timedelta(minutes=60)) is False

    def test_student_busy_naive_is_true(self, tz_url):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=2, student_id=1)
        repo = DatabaseRouter(tz_url).lessons

        busy_start = _naive_at(4, 10, 30)
        assert repo.check_student_conflict(1, busy_start, busy_start + timedelta(minutes=60)) is True

    def test_student_free_naive_is_false(self, tz_url):
        _add_lesson(tz_url, _naive_at(4, 10), teacher_id=2, student_id=1)
        repo = DatabaseRouter(tz_url).lessons

        free_start = _naive_at(4, 12)
        assert repo.check_student_conflict(1, free_start, free_start + timedelta(minutes=60)) is False


# ----------------------------------------------------------------------
# 5. 对话侧端到端：真 LessonChangeProcessor + 真 change_service
# ----------------------------------------------------------------------

class TestChatEndToEnd:

    def test_chat_reschedule_submits_request_without_raising(self, tz_url, svc):
        original = _naive_at(2)
        lid = _add_lesson(tz_url, original, teacher_id=1, student_id=1)
        _seed_neighbours(tz_url)
        lesson = _lesson(tz_url, lid)
        lesson["teacher_name"] = "张明"

        processor = LessonChangeProcessor.__new__(LessonChangeProcessor)
        processor.llm = None
        processor.message_builder = LessonChangeMessageBuilder()
        processor.change_service = svc

        data = {
            "new_time": _naive_at(3, 15).strftime("%Y-%m-%d %H:%M"),
            "scope": RescheduleMode.ONLY_THIS,
        }
        output = _run(_collect(processor._handle_reschedule(lesson, data)))

        assert "等待老师确认" in output
        assert "调课成功" not in output
        assert _lesson(tz_url, lid)["scheduled_at"] == original
        assert [row["status"] for row in svc.list_for_lesson(lid)] == ["pending"]


# ----------------------------------------------------------------------
# 6. HTTP 端到端：ISO 带偏移的 new_time
# ----------------------------------------------------------------------

class TestHttpEndToEnd:

    def test_reschedule_route_with_offset_iso_returns_200(self, point_db_to_tmp):
        from fastapi.testclient import TestClient
        from app import app

        url = point_db_to_tmp
        original = _naive_at(2)
        lid = _add_lesson(url, original, teacher_id=1, student_id=1)
        _seed_neighbours(url)
        client = TestClient(app)

        r = client.post(
            f"/api/lessons/{lid}/reschedule",
            json={"new_time": _aware(_naive_at(3, 15)).isoformat()},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "pending_teacher_confirm"
        assert _lesson(url, lid)["scheduled_at"] == original
