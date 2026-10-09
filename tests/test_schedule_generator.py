"""
ScheduleGenerator 单元测试

按 TUTOR_DESIGN.md §9.2 测试：
- 周模式解析
- 系列课生成（基础 + 跳过冲突）
- 时间格式兼容
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from agents.lesson.schedule_generator import ScheduleGenerator
from agents.lesson.conflict_detector import ConflictDetector, Conflict, ConflictType, ConflictSeverity


class TestWeeklyPatternParsing:
    """周模式解析测试"""

    def setup_method(self):
        self.gen = ScheduleGenerator()

    def test_parse_standard_format(self):
        """测试：标准格式 {day_of_week, time}"""
        raw = [{"day_of_week": 3, "time": "19:00"}]
        result = self.gen.parse_weekly_pattern(raw)
        assert result == [{"day_of_week": 3, "time": "19:00"}]

    def test_parse_alternative_format(self):
        """测试：兼容 {day_of_week, start_time}"""
        raw = [{"day_of_week": 3, "start_time": "19:00"}]
        result = self.gen.parse_weekly_pattern(raw)
        assert result == [{"day_of_week": 3, "time": "19:00"}]

    def test_parse_mixed_formats(self):
        """测试：混合格式"""
        raw = [
            {"day_of_week": 1, "start_time": "18:00"},
            {"day_of_week": 3, "time": "20:00"},
        ]
        result = self.gen.parse_weekly_pattern(raw)
        assert result == [
            {"day_of_week": 1, "time": "18:00"},
            {"day_of_week": 3, "time": "20:00"},
        ]

    def test_parse_invalid_input_skipped(self):
        """测试：缺字段被跳过"""
        raw = [
            {"day_of_week": 1},  # 缺 time
            {"time": "19:00"},   # 缺 day_of_week
            {"day_of_week": 3, "time": "20:00"},
        ]
        result = self.gen.parse_weekly_pattern(raw)
        assert len(result) == 1
        assert result[0]["day_of_week"] == 3

    def test_expand_pattern_with_chinese_weekdays(self):
        """测试：中文周名展开"""
        result = self.gen.expand_pattern_with_chinese_weekdays(["周三", "周五"], "19:00")
        assert len(result) == 2
        assert result[0] == {"day_of_week": 2, "time": "19:00"}
        assert result[1] == {"day_of_week": 4, "time": "19:00"}

    def test_expand_pattern_invalid_weekday_ignored(self):
        """测试：无效周名被忽略"""
        result = self.gen.expand_pattern_with_chinese_weekdays(["周三", "周九"], "19:00")
        assert len(result) == 1


class TestSeriesGeneration:
    """系列课生成测试"""

    def _make_mock_repo(self):
        """构造一个 mock 的 lesson repo（实例，不是类）"""
        repo = MagicMock()
        repo.check_teacher_conflict.return_value = False
        repo.check_student_conflict.return_value = False
        return repo

    def setup_method(self):
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo()
        self.gen = ScheduleGenerator(cd)

    def test_basic_4_week_series(self):
        """测试：4 周系列课，1 个时段"""
        # 禁用 skip_past 通过覆写 conflict_detector 的 check_series_lessons 调用
        cd = self.gen.conflict_detector
        original = cd.check_series_lessons
        cd.check_series_lessons = lambda **kw: original(**{**kw, "skip_past": False})

        result = self.gen.generate_series_lessons(
            teacher_id=1, student_id=1,
            start_date=date(2099, 6, 16),  # Wednesday
            total_weeks=4,
            weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],
        )
        cd.check_series_lessons = original
        assert result["lessons_count"] == 4
        assert result["skipped_count"] == 0
        assert result["conflicts"] == []

    def test_2_slots_per_week_8_lessons(self):
        """测试：每周 2 个时段，4 周 = 8 节"""
        cd = self.gen.conflict_detector
        original = cd.check_series_lessons
        cd.check_series_lessons = lambda **kw: original(**{**kw, "skip_past": False})

        result = self.gen.generate_series_lessons(
            teacher_id=1, student_id=1,
            start_date=date(2099, 6, 16),  # Wednesday
            total_weeks=4,
            weekly_pattern=[
                {"day_of_week": 1, "time": "19:00"},  # Tuesday
                {"day_of_week": 3, "time": "19:00"},  # Thursday
            ],
        )
        cd.check_series_lessons = original
        assert result["lessons_count"] == 8

    def test_series_index_starts_from_1(self):
        """测试：series_index 从 1 开始"""
        cd = self.gen.conflict_detector
        original = cd.check_series_lessons
        cd.check_series_lessons = lambda **kw: original(**{**kw, "skip_past": False})

        result = self.gen.generate_series_lessons(
            teacher_id=1, student_id=1,
            start_date=date(2099, 6, 16),
            total_weeks=3,
            weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],
        )
        cd.check_series_lessons = original
        indices = [l["series_index"] for l in result["lessons"]]
        assert indices == [1, 2, 3]

    def test_availability_mismatch_creates_conflicts(self):
        """测试：availability 不匹配产生冲突"""
        cd = ConflictDetector()
        cd._lesson_repo = self._make_mock_repo()
        gen = ScheduleGenerator(cd)
        original = cd.check_series_lessons
        cd.check_series_lessons = lambda **kw: original(**{**kw, "skip_past": False})

        result = gen.generate_series_lessons(
            teacher_id=1, student_id=1,
            start_date=date(2099, 6, 16),  # Wednesday
            total_weeks=2,
            weekly_pattern=[{"day_of_week": 2, "time": "19:00"}],
            teacher_availability=[
                # 只有周一可用，周三不可用
                {"day": "monday", "slots": ["19:00-20:00"]},
            ],
        )
        cd.check_series_lessons = original
        # 因为 Wednesday 不在 availability 中，所有课时应被跳过硬冲突
        assert result["lessons_count"] == 0
        assert result["skipped_count"] > 0
        # 应有 TEACHER_OFF 冲突
        assert any(c["conflict_type"] == "老师不可用" for c in result["conflicts"])


class TestHelperMethods:
    """辅助方法测试"""

    def setup_method(self):
        self.gen = ScheduleGenerator()

    def test_count_total_lessons(self):
        """测试：总课时数计算"""
        pattern = [{"day_of_week": i, "time": "19:00"} for i in range(3)]
        assert self.gen.count_total_lessons(8, pattern) == 24

    def test_find_next_monday_from_monday(self):
        """测试：从周一开始，下个周一是当天"""
        from datetime import date
        # 2099-06-15 是周一
        assert self.gen.find_next_monday(date(2099, 6, 15)) == date(2099, 6, 15)

    def test_find_next_monday_from_tuesday(self):
        """测试：从周二开始，下个周一是下周一"""
        # 2099-06-16 是周二
        assert self.gen.find_next_monday(date(2099, 6, 16)) == date(2099, 6, 22)

    def test_find_next_monday_from_sunday(self):
        """测试：从周日开始，下个周一是次日"""
        # 2099-06-21 是周日
        assert self.gen.find_next_monday(date(2099, 6, 21)) == date(2099, 6, 22)