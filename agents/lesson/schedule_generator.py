"""
周固定系列排课生成器

按 TUTOR_DESIGN.md §9.2 实现：
- 解析「每周三周五 19:00，连排 12 周」
- 生成 LessonSeries + N 条 Lesson 记录
- 集成 ConflictDetector 自动跳过冲突时段
"""

import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple

from .conflict_detector import ConflictDetector, Conflict, ConflictSeverity

logger = logging.getLogger(__name__)


class ScheduleGenerator:
    """系列排课生成器"""

    def __init__(self, conflict_detector: ConflictDetector = None):
        self.conflict_detector = conflict_detector or ConflictDetector()

    def generate_series_lessons(
        self,
        teacher_id: int,
        student_id: int,
        start_date: date,
        total_weeks: int,
        weekly_pattern: List[Dict[str, Any]],
        duration_minutes: int = 60,
        subject: str = None,
        package_id: int = None,
        teacher_availability: List[Dict[str, Any]] = None,
        exam_dates: List[date] = None,
        skip_hard_conflicts: bool = True,
    ) -> Dict[str, Any]:
        """
        生成系列课的所有 Lesson 记录（含冲突处理）

        Args:
            weekly_pattern: [{"day_of_week": 3, "time": "19:00"}, ...]
                           day_of_week: 0=周一, 6=周日
            start_date: 起始日期（必须是周一，方便按周推进）
            total_weeks: 总周数
            duration_minutes: 单次时长（默认 60 分钟）
            subject: 学科
            package_id: 关联套餐 ID
            teacher_availability: 老师 availability（如不传则不检查）
            exam_dates: 考试日列表（如不传则不检查）
            skip_hard_conflicts: 是否自动跳过硬冲突（软冲突只警告）

        Returns:
            {
                "lessons": [Lesson 字典列表, ...],
                "lessons_count": int,
                "skipped_count": int,
                "conflicts": [Conflict.to_dict(), ...],
                "warnings": [str, ...],
            }
        """
        # 1. 检测所有冲突
        lessons_info, conflicts = self.conflict_detector.check_series_lessons(
            teacher_id=teacher_id,
            student_id=student_id,
            weekly_pattern=weekly_pattern,
            start_date=start_date,
            total_weeks=total_weeks,
            duration_minutes=duration_minutes,
            teacher_availability=teacher_availability,
            exam_dates=exam_dates,
            skip_past=True,
        )

        # 2. 套餐溢出检查
        planned_count = len([l for l in lessons_info if not l.get("skipped")])
        if package_id:
            overflow = self.conflict_detector.check_series_overflow(package_id, planned_count)
            if overflow:
                conflicts.append(overflow)
                if overflow.severity == ConflictSeverity.HARD:
                    return {
                        "lessons": [],
                        "lessons_count": 0,
                        "skipped_count": 0,
                        "conflicts": [c.to_dict() for c in conflicts],
                        "warnings": [],
                        "error": overflow.message,
                    }

        # 3. 按策略处理冲突，生成 lesson 列表
        lessons = []
        warnings = []
        skipped = 0
        series_index = 1

        for lesson_info in lessons_info:
            if lesson_info.get("skipped"):
                skipped += 1
                continue

            hard_conflicts = [
                c for c in lesson_info["conflicts"]
                if c.severity == ConflictSeverity.HARD
            ]
            soft_conflicts = [
                c for c in lesson_info["conflicts"]
                if c.severity == ConflictSeverity.SOFT
            ]

            # 硬冲突：默认跳过
            if hard_conflicts and skip_hard_conflicts:
                skipped += 1
                warnings.append(
                    f"第 {lesson_info['week']} 周 {lesson_info['scheduled_at'].strftime('%Y-%m-%d %H:%M')} "
                    f"被跳过：{hard_conflicts[0].message}"
                )
                continue

            # 软冲突：记录警告但仍排课
            for sc in soft_conflicts:
                warnings.append(
                    f"⚠️ 第 {lesson_info['week']} 周 {lesson_info['scheduled_at'].strftime('%Y-%m-%d %H:%M')}"
                    f"：{sc.message}"
                )

            lessons.append({
                "teacher_id": teacher_id,
                "student_id": student_id,
                "scheduled_at": lesson_info["scheduled_at"],
                "lesson_type": "series",
                "subject": subject,
                "duration_minutes": duration_minutes,
                "status": "confirmed",
                "series_index": series_index,
                "package_id": package_id,
            })
            series_index += 1

        return {
            "lessons": lessons,
            "lessons_count": len(lessons),
            "skipped_count": skipped,
            "conflicts": [c.to_dict() for c in conflicts],
            "warnings": warnings,
        }

    def parse_weekly_pattern(self, raw_pattern: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        规整化周模式输入
        - 兼容 {"day_of_week": 3, "start_time": "19:00"} 和 {"day_of_week": 3, "time": "19:00"} 两种格式
        - 输出统一格式 {"day_of_week": 3, "time": "19:00"}
        """
        normalized = []
        for p in raw_pattern:
            dow = p.get("day_of_week")
            time_str = p.get("time") or p.get("start_time")
            if dow is None or time_str is None:
                continue
            normalized.append({"day_of_week": int(dow), "time": str(time_str)})
        return normalized

    def expand_pattern_with_chinese_weekdays(
        self,
        weekday_names: List[str],
        time_str: str,
    ) -> List[Dict[str, Any]]:
        """
        辅助：从中文星期名生成 pattern
        weekday_names: ["周三", "周五"]
        time_str: "19:00"
        """
        weekday_map = {
            "周一": 0, "周二": 1, "周三": 2, "周四": 3,
            "周五": 4, "周六": 5, "周日": 6,
        }
        pattern = []
        for name in weekday_names:
            dow = weekday_map.get(name)
            if dow is not None:
                pattern.append({"day_of_week": dow, "time": time_str})
        return pattern

    def find_next_monday(self, from_date: date = None) -> date:
        """找到 from_date 之后的下一个周一（包含当天如果是周一）"""
        if from_date is None:
            from_date = date.today()
        days_ahead = 0 - from_date.weekday()  # 周一是 0
        if days_ahead < 0:
            days_ahead += 7
        return from_date + timedelta(days=days_ahead)

    def count_total_lessons(
        self,
        total_weeks: int,
        weekly_pattern: List[Dict[str, Any]],
    ) -> int:
        """计算系列课总课时数"""
        return total_weeks * len(weekly_pattern)