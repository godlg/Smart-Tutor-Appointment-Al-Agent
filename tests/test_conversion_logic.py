"""
ConversionLogic 单元测试

按 TUTOR_DESIGN.md §5.3 / §9.1 测试：
- 流失原因分类
- 套餐评分逻辑
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.trial.conversion_logic import (
    ConversionLogic,
    ConversionStrategy,
    ConversionOutcome,
)


class TestChurnReasonClassification:
    """流失原因分类测试"""

    def setup_method(self):
        # 不依赖 DB 的实例化
        self.logic = ConversionLogic(
            package_service=MagicMock(),
            teacher_service=MagicMock(),
            student_service=MagicMock(),
        )

    @pytest.mark.parametrize("rating,text,expected", [
        (1, "价格太贵了", "price"),
        (1, "老师风格不适合", "style_mismatch"),
        (1, "上课时间不方便", "schedule"),
        (3, "暂不需要了", "no_need"),
        (5, "很好", "not_churn"),  # would_continue=True
        (4, "", "other"),
        (3, "老师讲得太快了", "other"),
        (1, "教学太严厉了", "style_mismatch"),
    ])
    def test_classify_churn_reason(self, rating, text, expected):
        would_continue = text == "很好"
        result = self.logic.classify_churn_reason({
            "would_continue": would_continue,
            "rating": rating,
            "feedback_text": text,
        })
        assert result == expected


class TestPackageScoring:
    """套餐评分测试"""

    def setup_method(self):
        self.logic = ConversionLogic(
            package_service=MagicMock(),
            teacher_service=MagicMock(),
            student_service=MagicMock(),
        )

    def test_subject_match_adds_significant_score(self):
        """测试：学科匹配加分最多"""
        pkg = {"subject": "数学", "grade_level": "高中", "total_lessons": 12, "price_total": 3000.0}
        student = {"weak_subjects": ["数学", "物理"], "grade_level": "高中"}

        score = self.logic._score_package(pkg, student)
        assert score >= 70  # 学科 40 + 年级 30 + 价格 20

    def test_no_subject_match_no_high_score(self):
        """测试：学科不匹配得不到高分"""
        pkg = {"subject": "语文", "grade_level": "高中", "total_lessons": 12, "price_total": 3000.0}
        student = {"weak_subjects": ["数学"], "grade_level": "高中"}

        score = self.logic._score_package(pkg, student)
        # 应该低于有学科匹配的情况
        assert score < 70

    def test_high_lesson_count_bonus_for_multiple_weak_subjects(self):
        """测试：多薄弱学科 + 长课时组合加分"""
        pkg = {"subject": "数学", "grade_level": "高中", "total_lessons": 24, "price_total": 5000.0}
        student = {"weak_subjects": ["数学", "物理", "化学"], "grade_level": "高中"}

        score = self.logic._score_package(pkg, student)
        # 学科 40 + 年级 30 + 多薄弱 15 = 85（价格档超出默认预算 *6）
        assert score >= 80

    def test_match_reason_contains_subject(self):
        """测试：匹配理由包含学科"""
        pkg = {"subject": "数学", "grade_level": "高中"}
        student = {"weak_subjects": ["数学"], "grade_level": "高中"}
        reason = self.logic._build_match_reason(pkg, student)
        assert "数学" in reason


class TestConversionOutcomes:
    """转化结果枚举测试"""

    def test_outcome_values(self):
        assert ConversionOutcome.CONVERTED == "converted"
        assert ConversionOutcome.PENDING == "pending"
        assert ConversionOutcome.CHURNED == "churned"
        assert ConversionOutcome.REMIND_LATER == "remind_later"