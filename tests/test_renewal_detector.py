"""
RenewalDetector 单元测试

按 TUTOR_DESIGN.md §5.5 / §9.4 测试：
- 续费触发阈值
- 套餐推荐排序
"""

import pytest
from unittest.mock import MagicMock

from agents.user_behavior.renewal_detector import (
    RenewalDetector,
    LOW_LESSONS_THRESHOLD,
    NEAR_EXPIRY_DAYS,
)


class TestRenewalThresholds:
    """续费阈值常量测试"""

    def test_low_lessons_threshold_is_reasonable(self):
        assert LOW_LESSONS_THRESHOLD == 3

    def test_near_expiry_days_is_reasonable(self):
        assert NEAR_EXPIRY_DAYS == 14


class TestShouldTriggerRenewal:
    """续费触发检测测试"""

    def test_no_series_returns_no_trigger(self):
        """测试：无系列课时不触发"""
        rd = RenewalDetector(
            student_service=MagicMock(),
        )
        # Mock DB
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            mock_db.lessons.get_series_by_student.return_value = []

            result = rd.should_trigger_renewal(student_id=1)
            assert result["should_trigger"] is False

    def test_low_remaining_lessons_triggers(self):
        """测试：剩余课时不足触发"""
        rd = RenewalDetector(student_service=MagicMock())
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            from datetime import date
            mock_db.lessons.get_series_by_student.return_value = [{
                "id": 1,
                "total_lessons": 24,
                "completed_lessons": 22,
                "cancelled_lessons": 0,
                "end_date": date.today().replace(year=date.today().year + 1),
                "package_id": 1,
                "created_at": date.today(),
            }]
            result = rd.should_trigger_renewal(student_id=1)
            assert result["should_trigger"] is True
            assert "剩 2" in result["reason"]
            assert result["urgency"] in ("medium", "high")

    def test_very_low_remaining_is_high_urgency(self):
        """测试：剩余 0-1 课时 = 高紧迫度"""
        rd = RenewalDetector(student_service=MagicMock())
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            from datetime import date
            mock_db.lessons.get_series_by_student.return_value = [{
                "id": 1,
                "total_lessons": 24,
                "completed_lessons": 24,
                "cancelled_lessons": 0,
                "end_date": date.today().replace(year=date.today().year + 1),
                "package_id": 1,
                "created_at": date.today(),
            }]
            result = rd.should_trigger_renewal(student_id=1)
            assert result["remaining_lessons"] == 0
            assert result["urgency"] == "high"

    def test_near_expiry_triggers(self):
        """测试：临近过期触发"""
        rd = RenewalDetector(student_service=MagicMock())
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            from datetime import date, timedelta
            mock_db.lessons.get_series_by_student.return_value = [{
                "id": 1,
                "total_lessons": 24,
                "completed_lessons": 5,
                "cancelled_lessons": 0,
                "end_date": date.today() + timedelta(days=5),  # 5 天后过期
                "package_id": 1,
                "created_at": date.today() - timedelta(days=180),
            }]
            result = rd.should_trigger_renewal(student_id=1)
            assert result["should_trigger"] is True
            assert "过期" in result["reason"]


class TestRenewalMessageGeneration:
    """续费消息生成测试"""

    def test_no_trigger_returns_empty(self):
        rd = RenewalDetector(student_service=MagicMock())
        message = rd.generate_renewal_message(
            student_id=1,
            trigger_info={"should_trigger": False, "reason": "暂不需要续费"},
        )
        assert message == ""


class TestRenewalPackageRecommendation:
    """续费套餐推荐测试"""

    def test_filter_out_current_package(self):
        """测试：过滤掉当前已购套餐"""
        package_svc = MagicMock()
        package_svc.get_package.return_value = {"teacher_id": 1, "name": "当前套餐"}
        package_svc.get_all_packages.return_value = [
            {"id": 1, "name": "当前套餐", "total_lessons": 12, "price_total": 3000.0},
            {"id": 2, "name": "续费套餐A", "total_lessons": 24, "price_total": 5000.0},
            {"id": 3, "name": "续费套餐B", "total_lessons": 36, "price_total": 7000.0},
        ]
        student_svc = MagicMock()
        student_svc.get_student.return_value = {
            "id": 1, "name": "小明", "weak_subjects": ["数学"], "grade_level": "高中",
        }

        rd = RenewalDetector(
            package_service=package_svc,
            student_service=student_svc,
        )
        result = rd.recommend_renewal_packages(
            student_id=1, current_package_id=1, limit=5,
        )
        # 不应包含 ID=1 的当前套餐
        assert all(p["id"] != 1 for p in result)


# Patch the DatabaseRouter reference
from unittest.mock import patch