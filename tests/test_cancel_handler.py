"""
CancelHandler 单元测试

按 TUTOR_DESIGN.md §9 测试：
- 单次课取消 + 退费计算
- 系列课取消
- 退费政策梯度
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agents.lesson.cancel_handler import (
    CancelHandler,
    REFUND_POLICY,
)


class TestRefundCalculation:
    """退费计算测试"""

    def setup_method(self):
        self.handler = CancelHandler()

    def test_free_lesson_no_refund_needed(self):
        """测试：免费课时不涉及退费"""
        info = self.handler.calculate_refund(
            {"price": 0.0, "scheduled_at": datetime.now() + timedelta(hours=72)}
        )
        assert info["refund_amount"] == 0.0
        assert info["refund_ratio"] == 1.0

    def test_72_hours_advance_full_refund(self):
        """测试：72h 前取消 = 100% 退款"""
        info = self.handler.calculate_refund(
            {"price": 300.0, "scheduled_at": datetime.now() + timedelta(hours=72)}
        )
        assert info["refund_amount"] == 300.0
        assert info["refund_ratio"] == 1.0

    def test_36_hours_advance_80_percent_refund(self):
        """测试：36h 前取消 = 80% 退款"""
        info = self.handler.calculate_refund(
            {"price": 300.0, "scheduled_at": datetime.now() + timedelta(hours=36)}
        )
        assert info["refund_amount"] == 240.0
        assert info["refund_ratio"] == 0.8

    def test_18_hours_advance_50_percent_refund(self):
        """测试：18h 前取消 = 50% 退款"""
        info = self.handler.calculate_refund(
            {"price": 300.0, "scheduled_at": datetime.now() + timedelta(hours=18)}
        )
        assert info["refund_amount"] == 150.0
        assert info["refund_ratio"] == 0.5

    def test_6_hours_advance_no_refund(self):
        """测试：6h 前取消 = 0% 退款"""
        info = self.handler.calculate_refund(
            {"price": 300.0, "scheduled_at": datetime.now() + timedelta(hours=6)}
        )
        assert info["refund_amount"] == 0.0
        assert info["refund_ratio"] == 0.0

    def test_teacher_reason_gives_full_refund(self):
        """测试：老师原因应全额退款"""
        info = self.handler.calculate_refund(
            {"price": 300.0, "scheduled_at": datetime.now() + timedelta(hours=6)},
            reason="老师档期问题，无法上课",
        )
        assert info["refund_amount"] == 300.0
        assert info["refund_ratio"] == 1.0
        assert "全额" in info["policy"]

    def test_policy_threshold_progression(self):
        """测试：退费梯度按预期工作"""
        test_cases = [
            (timedelta(hours=100), 1.0),  # 48h+: 100%
            (timedelta(hours=48), 1.0),   # 48h: 100%
            (timedelta(hours=36), 0.8),   # 24-48h: 80%
            (timedelta(hours=24), 0.8),   # 24h: 80%
            (timedelta(hours=18), 0.5),   # 12-24h: 50%
            (timedelta(hours=12), 0.5),   # 12h: 50%
            (timedelta(hours=6), 0.0),    # 不到 12h: 0%
            (timedelta(hours=0), 0.0),    # 已开始: 0%
            (timedelta(hours=-2), 0.0),   # 已过去: 0%
        ]
        for delta, expected_ratio in test_cases:
            info = self.handler.calculate_refund(
                {"price": 200.0, "scheduled_at": datetime.now() + delta}
            )
            assert info["refund_ratio"] == expected_ratio, \
                f"Failed for delta={delta}: expected {expected_ratio}, got {info['refund_ratio']}"


class TestCancelLesson:
    """取消单次课时测试"""

    def setup_method(self):
        self.handler = CancelHandler()

    def test_cancel_nonexistent_lesson_returns_false(self):
        """测试：取消不存在的课时"""
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            mock_db.lessons.get_lesson_by_id.return_value = None

            ok, msg, refund = self.handler.cancel_lesson(lesson_id=999)
            assert ok is False
            assert "不存在" in msg
            assert refund is None

    def test_cancel_completed_lesson_returns_false(self):
        """测试：取消已完成的课时被拒绝"""
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            mock_db.lessons.get_lesson_by_id.return_value = {
                "id": 1, "status": "completed", "price": 300.0,
                "scheduled_at": datetime.now() + timedelta(hours=48),
                "series_id": None,
            }
            ok, msg, _ = self.handler.cancel_lesson(lesson_id=1)
            assert ok is False
            assert "已完成" in msg

    def test_cancel_already_cancelled_returns_false(self):
        """测试：取消已取消的课时被拒绝"""
        with patch("db.db_router.DatabaseRouter") as MockDB:
            mock_db = MockDB.return_value
            mock_db.lessons.get_lesson_by_id.return_value = {
                "id": 1, "status": "cancelled", "price": 300.0,
                "scheduled_at": datetime.now() + timedelta(hours=48),
                "series_id": None,
            }
            ok, msg, _ = self.handler.cancel_lesson(lesson_id=1)
            assert ok is False


class TestRefundPolicy:
    """退费政策常量测试"""

    def test_policy_is_descending(self):
        """测试：退费政策随时间临近单调下降"""
        ratios = [r for _, r in REFUND_POLICY]
        for i in range(len(ratios) - 1):
            # 第一个元素（最长提前期）应该是较高比例
            assert ratios[i] >= ratios[i + 1], \
                f"Ratio should descend: {ratios[i]} >= {ratios[i+1]}"

    def test_policy_has_full_refund_tier(self):
        """测试：政策包含全额退款档"""
        assert any(r == 1.0 for _, r in REFUND_POLICY)

    def test_policy_has_zero_refund_tier(self):
        """测试：政策包含零退款档"""
        assert any(r == 0.0 for _, r in REFUND_POLICY)