"""
P0 / P1 回归测试：退课 REST 接口被重复路由遮蔽 + 系列课空生成静默成功

背景：
- P0: api/lesson.py 曾有两个 @router.post("/{lesson_id}/cancel")，第一个（只返回
      {"success": bool}）遮蔽了第二个（CancelHandler + 退费 + 原因落库）。
      修复后响应必须含 refund 与 message 字段，reason 写入 parent_feedback。
- P1: services/lesson_service.py create_lesson_series 在所有课时被跳过时
      曾无条件返回 (True, {...}) 并留下 total_lessons=0 的空系列。
      修复后应返回 (False, {"error": "系列课全部课时被跳过", ...})，且不落库。

测试用 mock 隔离真实 DB（沿用 test_cancel_handler.py 的 patch 惯例）。
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch, MagicMock  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


# ---------------- P0：REST 退课返回 refund + message ----------------

class TestCancelLessonEndpoint:
    """POST /api/lessons/{id}/cancel 必须走 CancelHandler（含退费），而非被遮蔽的简单版。"""

    def _mock_db(self, lesson):
        """构造 mock DB：CancelHandler 从 db.db_router 导入 DatabaseRouter。"""
        mock_db = MagicMock()
        mock_db.lessons.get_lesson_by_id.return_value = lesson
        mock_db.lessons.update_lesson.return_value = True
        return mock_db

    def test_cancel_returns_refund_and_message(self, client):
        """响应必须包含 refund 与 message（不止 success）"""
        lesson = {
            "id": 1, "status": "pending", "price": 300.0,
            "scheduled_at": datetime.now() + timedelta(hours=72),
            "series_id": None,
        }
        with patch("db.db_router.DatabaseRouter") as MockDB:
            MockDB.return_value = self._mock_db(lesson)
            r = client.post("/api/lessons/1/cancel", json={"reason": "家长时间冲突"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert "message" in body
        assert "refund" in body
        refund = body["refund"]
        assert "refund_amount" in refund
        assert "refund_ratio" in refund
        assert "policy" in refund
        # 72h 前退课 → 100% 退费
        assert refund["refund_amount"] == 300.0
        assert refund["refund_ratio"] == 1.0

    def test_cancel_reason_written_to_parent_feedback(self, client):
        """reason 必须随退课写入 parent_feedback"""
        lesson = {
            "id": 2, "status": "pending", "price": 200.0,
            "scheduled_at": datetime.now() + timedelta(hours=72),
            "series_id": None,
        }
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = self._mock_db(lesson)
            MockDB.return_value = mock_db
            r = client.post("/api/lessons/2/cancel", json={"reason": "孩子生病请假"})
        assert r.status_code == 200, r.text
        assert r.json()["success"] is True
        # 断言 update_lesson 被调用且 parent_feedback 携带退课原因
        call_kwargs = mock_db.lessons.update_lesson.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs["status"] == "cancelled"
        assert kwargs["parent_feedback"] == "[退课原因] 孩子生病请假"

    def test_cancel_without_reason_still_returns_refund(self, client):
        """无 reason 时响应仍含 refund 与 message"""
        lesson = {
            "id": 3, "status": "pending", "price": 300.0,
            "scheduled_at": datetime.now() + timedelta(hours=72),
            "series_id": None,
        }
        with patch("db.db_router.DatabaseRouter") as MockDB:
            MockDB.return_value = self._mock_db(lesson)
            r = client.post("/api/lessons/3/cancel", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert "refund" in body
        assert "message" in body


# ---------------- P1：空系列生成不得落库 ----------------

class TestCreateSeriesAllSkipped:
    """create_lesson_series 在所有课时被跳过时应返回 False 且不创建系列。"""

    def _make_svc_with_mock_db(self):
        """patch DatabaseRouter 构造隔离 DB 的 LessonService。"""
        from services.lesson_service import LessonService
        svc = LessonService()
        mock_db = MagicMock()
        svc.db = mock_db
        return svc, mock_db

    def test_all_skipped_returns_false_and_no_series_record(self):
        from services.lesson_service import LessonService

        gen_result = {
            "lessons": [],
            "lessons_count": 0,
            "skipped_count": 4,
            "conflicts": [{"severity": "hard", "message": "老师该时段不可用", "conflict_type": "老师不可用"}],
            "warnings": [],
        }

        svc = LessonService()
        mock_db = MagicMock()
        svc.db = mock_db

        with patch("agents.lesson.ScheduleGenerator") as MockGen:
            MockGen.return_value.parse_weekly_pattern.side_effect = lambda p: p
            MockGen.return_value.generate_series_lessons.return_value = gen_result

            ok, result = svc.create_lesson_series(
                teacher_id=1, student_id=1,
                weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],
                start_date=datetime.now().date(),
                total_weeks=4,
            )

        assert ok is False
        assert result["error"] == "系列课全部课时被跳过"
        assert "skipped" in result
        assert len(result["skipped"]) >= 1
        # 不得创建系列记录
        mock_db.lessons.add_series.assert_not_called()
        mock_db.lessons.add_lesson.assert_not_called()

    def test_partial_generation_still_creates_series(self):
        """部分课时被跳过（但非全部）时仍正常创建系列——回归保护。"""
        from services.lesson_service import LessonService

        gen_result = {
            "lessons": [
                {"teacher_id": 1, "student_id": 1, "scheduled_at": datetime.now(),
                 "subject": "数学", "duration_minutes": 60, "series_index": 1, "package_id": None},
            ],
            "lessons_count": 1,
            "skipped_count": 1,
            "conflicts": [],
            "warnings": [],
        }

        svc = LessonService()
        mock_db = MagicMock()
        svc.db = mock_db
        mock_db.lessons.add_series.return_value = 100

        with patch("agents.lesson.ScheduleGenerator") as MockGen:
            MockGen.return_value.parse_weekly_pattern.side_effect = lambda p: p
            MockGen.return_value.generate_series_lessons.return_value = gen_result

            ok, result = svc.create_lesson_series(
                teacher_id=1, student_id=1,
                weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],
                start_date=datetime.now().date(),
                total_weeks=4,
            )

        assert ok is True
        assert result["series_id"] == 100
        assert result["lessons_created"] == 1
        mock_db.lessons.add_series.assert_called_once()
        mock_db.lessons.add_lesson.assert_called_once()
