"""调课需老师确认：对话侧与 HTTP 侧统一走调课请求（C2c）

覆盖：
1-3. 对话侧：合法调课只提交申请（不改期、不说「调课成功」）；幂等命中提示已提交；
     LessonChangeError 把 detail 带给家长
4-6. HTTP：默认模式只建 pending 请求（课时时间不变）；immediate=true 立即改期且返回形状不变；
     默认模式冲突返回 409 而不是 500

约定：对话侧 change_service 用 MagicMock 替身；HTTP 用例走 TestClient，
并把 LessonChangeService 默认库与 RescheduleHandler 内部的 DatabaseRouter 一并指向 tmp_path 临时库，
immediate=true 会真改课时，绝不能落到生产库。不硬编码日期（datetime.now 相对计算）。
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from agents.lesson.change_processor import LessonChangeMessageBuilder, LessonChangeProcessor
from agents.lesson.reschedule_handler import RescheduleMode
from db import db_router as db_router_module
from db.base.session_manager import SessionManager
from db.db_router import DatabaseRouter
from db.models import StudentProfile, Teacher
from services.lesson_change_service import LessonChangeError, LessonChangeService


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream) -> str:
    tokens = []
    async for token in stream:
        tokens.append(token)
    return "".join(tokens)


def _at(days: int, hour: int = 10, minute: int = 0) -> datetime:
    """相对今天的时刻（不硬编码日期）。"""
    base = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)


def _make_lesson(lesson_id=101, days_ahead=2):
    return {
        "id": lesson_id,
        "teacher_id": 1,
        "student_id": 1,
        "series_id": None,
        "subject": "数学",
        "scheduled_at": datetime.now() + timedelta(days=days_ahead),
        "duration_minutes": 60,
        "status": "confirmed",
        "price": 200.0,
    }


def _created_result(mode=RescheduleMode.ONLY_THIS):
    return {
        "created": True,
        "request": {
            "id": 7,
            "mode": mode,
            "new_scheduled_at": datetime(2030, 6, 7, 16, 0),
            "status": "pending",
        },
    }


# ----------------------------------------------------------------------
# 1-3. 对话侧
# ----------------------------------------------------------------------

def _build_chat_processor(lesson, parsed_data):
    """不碰 LLM / 数据库的对话处理器（与 test_reschedule_routing 同一做法）。"""

    class _FakeParser:
        async def parse(self, user_input, lessons_text, intent):
            return parsed_data

    processor = LessonChangeProcessor.__new__(LessonChangeProcessor)
    processor.llm = MagicMock()
    processor.parser = _FakeParser()
    processor.message_builder = LessonChangeMessageBuilder()
    processor.change_service = MagicMock()
    processor.cancel_handler = MagicMock()

    processor.student_service = MagicMock()
    processor.student_service.initialize_default_student.return_value = {"id": 1}
    processor.lesson_service = MagicMock()
    processor.lesson_service.get_student_upcoming_lessons.return_value = [lesson]
    processor.teacher_service = MagicMock()
    processor.teacher_service.get_teacher_by_id.return_value = {"name": "张老师"}
    return processor


def _reschedule_data(scope=RescheduleMode.ONLY_THIS):
    return {
        "target_lesson_id": 101,
        "new_time": "2030-06-07 16:00",
        "scope": scope,
        "reason": "未知",
    }


class TestChatReschedule:

    def test_legal_reschedule_submits_request_not_success(self):
        # Arrange
        processor = _build_chat_processor(_make_lesson(), _reschedule_data())
        processor.change_service.create_request.return_value = _created_result()

        # Act
        output = _run(_collect(
            processor.handle_change_request("把周三的课改到周五下午4点", "reschedule")
        ))

        # Assert
        processor.change_service.create_request.assert_called_once_with(
            lesson_id=101,
            new_time=datetime(2030, 6, 7, 16, 0, tzinfo=timezone(timedelta(hours=8))),
            mode=RescheduleMode.ONLY_THIS,
            requested_by="parent",
        )
        assert "等待老师确认" in output
        assert "调课成功" not in output

    def test_duplicate_request_says_already_submitted(self):
        # Arrange
        processor = _build_chat_processor(_make_lesson(), _reschedule_data())
        processor.change_service.create_request.return_value = {
            "created": False,
            "request": {
                "id": 7,
                "mode": RescheduleMode.ONLY_THIS,
                "new_scheduled_at": datetime(2030, 6, 7, 16, 0),
                "status": "pending",
            },
        }

        # Act
        output = _run(_collect(
            processor.handle_change_request("改到周五下午4点", "reschedule")
        ))

        # Assert
        assert "已经提交过" in output
        assert "等待老师确认" in output
        assert "调课成功" not in output

    def test_service_error_detail_is_passed_to_parent(self):
        # Arrange
        processor = _build_chat_processor(_make_lesson(), _reschedule_data())
        processor.change_service.create_request.side_effect = LessonChangeError(
            "新时间与老师已有课程冲突", 409,
        )

        # Act
        output = _run(_collect(
            processor.handle_change_request("改到周五下午4点", "reschedule")
        ))

        # Assert
        assert "没有成功" in output
        assert "新时间与老师已有课程冲突" in output
        assert "调课成功" not in output


# ----------------------------------------------------------------------
# 4-6. HTTP 侧（TestClient + 临时库）
# ----------------------------------------------------------------------

OPEN_AVAILABILITY = [
    {"day": day, "slots": ["00:00-23:59"]}
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
]


@pytest.fixture
def chat_url(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'chat_flow.db').as_posix()}"


@pytest.fixture
def http_env(monkeypatch, chat_url):
    """老师 1 + 两个学生；调课服务与 RescheduleHandler 的库都指向临时库。"""
    from fastapi.testclient import TestClient
    from app import app
    from services import lesson_change_service as change_module

    sm = SessionManager(chat_url)
    try:
        with sm.session_scope() as session:
            session.add(Teacher(id=1, name="张明", is_active=True, availability=OPEN_AVAILABILITY))
            session.add(StudentProfile(name="学生甲"))
            session.add(StudentProfile(name="学生乙"))
    finally:
        sm.close()

    monkeypatch.setattr(change_module, "DEFAULT_DB_PATH", chat_url)
    real_router = db_router_module.DatabaseRouter
    monkeypatch.setattr(db_router_module, "DatabaseRouter", lambda *a, **k: real_router(chat_url))
    return {"client": TestClient(app), "url": chat_url}


def _add_lesson(url: str, **kwargs) -> int:
    defaults = dict(
        lesson_type="regular", subject="数学", duration_minutes=60,
        status="confirmed", series_id=None, series_index=None,
        package_id=None, price=None,
    )
    defaults.update(kwargs)
    return DatabaseRouter(url).lessons.add_lesson(**defaults)


def _lesson_time(url: str, lesson_id: int):
    return DatabaseRouter(url).lessons.get_lesson_by_id(lesson_id)["scheduled_at"]


class TestHttpReschedule:

    def test_default_mode_creates_pending_request_and_keeps_time(self, http_env):
        url = http_env["url"]
        original = _at(2)
        lid = _add_lesson(url, teacher_id=1, student_id=1, scheduled_at=original)

        r = http_env["client"].post(
            f"/api/lessons/{lid}/reschedule", json={"new_time": _at(3, 15).isoformat()},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert body["status"] == "pending_teacher_confirm"
        assert body["created"] is True
        assert isinstance(body["request_id"], int)
        assert "message" in body
        assert _lesson_time(url, lid) == original
        pending = LessonChangeService(db_path=url)
        try:
            rows = pending.list_for_lesson(lid)
        finally:
            pending.close()
        assert [row["status"] for row in rows] == ["pending"]

    def test_immediate_true_reschedules_now(self, http_env):
        url = http_env["url"]
        lid = _add_lesson(url, teacher_id=1, student_id=1, scheduled_at=_at(2))
        new_time = _at(3, 15)

        r = http_env["client"].post(
            f"/api/lessons/{lid}/reschedule",
            json={"new_time": new_time.isoformat(), "immediate": True},
        )

        assert r.status_code == 200, r.text
        assert set(r.json().keys()) == {"success", "message"}
        assert _lesson_time(url, lid) == new_time

    def test_default_mode_conflict_is_409_and_keeps_time(self, http_env):
        url = http_env["url"]
        original = _at(2)
        lid = _add_lesson(url, teacher_id=1, student_id=1, scheduled_at=original)
        busy = _at(4, 10)
        _add_lesson(url, teacher_id=1, student_id=2, scheduled_at=busy)

        r = http_env["client"].post(
            f"/api/lessons/{lid}/reschedule", json={"new_time": busy.isoformat()},
        )

        assert r.status_code == 409, r.text
        assert _lesson_time(url, lid) == original
