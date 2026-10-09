"""
ConflictDetector 单元测试

按 TUTOR_DESIGN.md §5.4 测试各类冲突检测：
- STUDENT_BUSY
- TEACHER_BUSY
- TEACHER_OFF
- EXAM_PERIOD
- BLACKOUT_DATE
- SERIES_OVERFLOW
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from agents.lesson.conflict_detector import (
    ConflictDetector,
    Conflict,
    ConflictType,
    ConflictSeverity,
)


class TestConflictDetectorBasics:
    """基础冲突检测测试"""

    def _make_mock_repo(self, teacher_conflict=False, student_conflict=False):
        repo = MagicMock()
        repo.check_teacher_conflict.return_value = teacher_conflict
        repo.check_student_conflict.return_value = student_conflict
        return repo

    def test_no_conflict_returns_empty_list(self):
        """测试：无冲突时返回空列表"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo()

        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 15, 19, 0),
            duration_minutes=60,
        )
        assert len(conflicts) == 0

    def test_teacher_busy_creates_hard_conflict(self):
        """测试：老师档期冲突生成硬冲突"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo(teacher_conflict=True)

        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 15, 19, 0),
            duration_minutes=60,
        )
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.TEACHER_BUSY
        assert conflicts[0].severity == ConflictSeverity.HARD

    def test_student_busy_creates_hard_conflict(self):
        """测试：学生档期冲突"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo(student_conflict=True)

        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 15, 19, 0),
            duration_minutes=60,
        )
        assert any(c.conflict_type == ConflictType.STUDENT_BUSY for c in conflicts)

    def test_teacher_unavailable_outside_availability(self):
        """测试：老师在 availability 之外"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo()

        availability = [
            {"day": "monday", "slots": ["19:00-20:00"]},
        ]
        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 17, 19, 0),  # Wednesday
            duration_minutes=60,
            teacher_availability=availability,
        )
        assert any(c.conflict_type == ConflictType.TEACHER_OFF for c in conflicts)

    def test_exam_period_is_soft_conflict(self):
        """测试：考试日是软冲突"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo()

        exam_date = date(2099, 6, 15)
        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 12, 19, 0),
            duration_minutes=60,
            exam_dates=[exam_date],
        )
        exam_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.EXAM_PERIOD]
        assert len(exam_conflicts) == 1
        assert exam_conflicts[0].severity == ConflictSeverity.SOFT

    def test_blackout_date_is_soft_conflict(self):
        """测试：黑名单日是软冲突"""
        cd = ConflictDetector(blackout_dates=["2099-06-15"])
        cd._lesson_repo = self._make_mock_repo()

        conflicts = cd.check_single_lesson(
            teacher_id=1,
            student_id=1,
            scheduled_at=datetime(2099, 6, 15, 19, 0),
            duration_minutes=60,
        )
        blackout = [c for c in conflicts if c.conflict_type == ConflictType.BLACKOUT_DATE]
        assert len(blackout) == 1
        assert blackout[0].severity == ConflictSeverity.SOFT


class TestConflictDetectorSeries:
    """系列课冲突检测"""

    def test_series_check_returns_per_lesson_info(self):
        """测试：系列课检测返回每节详情"""
        cd = ConflictDetector()
        cd._lesson_repo = MagicMock()
        cd._lesson_repo.check_teacher_conflict.return_value = False
        cd._lesson_repo.check_student_conflict.return_value = False

        lessons_info, all_conflicts = cd.check_series_lessons(
            teacher_id=1,
            student_id=1,
            weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],  # Wednesday
            start_date=date(2099, 6, 16),  # Wednesday
            total_weeks=4,
            duration_minutes=60,
            teacher_availability=[
                {"day": "wednesday", "slots": ["19:00-20:00"]},
            ],
            skip_past=False,  # 测试需要：不要跳过过去日期
        )
        # 4 周 × 1 个时段 = 4 节
        assert len(lessons_info) == 4


class TestConflictToDict:
    """Conflict.to_dict 测试"""

    def test_to_dict_includes_all_fields(self):
        c = Conflict(
            conflict_type=ConflictType.TEACHER_BUSY,
            severity=ConflictSeverity.HARD,
            message="测试",
            related_id=42,
            related_date=date(2099, 6, 15),
            extra={"key": "value"},
        )
        d = c.to_dict()
        assert d["conflict_type"] == "老师已被预约"
        assert d["severity"] == "hard"
        assert d["message"] == "测试"
        assert d["related_id"] == 42
        assert d["related_date"] == "2099-06-15"
        assert d["extra"] == {"key": "value"}