"""
冲突检测器

按 TUTOR_DESIGN.md §5.4 定义，检测以下冲突：
- STUDENT_BUSY: 学生已有课程
- TEACHER_BUSY: 老师已被预约
- TEACHER_OFF: 老师不在 availability 时段
- EXAM_PERIOD: 考试日前后禁课
- BLACKOUT_DATE: 平台黑名单日（节假日等）
- SERIES_OVERFLOW: 超出套餐总课时
"""

import logging
from enum import Enum
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """冲突类型"""
    STUDENT_BUSY = "学生已有课程"
    TEACHER_BUSY = "老师已被预约"
    TEACHER_OFF = "老师不可用"
    EXAM_PERIOD = "考试日前后禁课"
    BLACKOUT_DATE = "平台黑名单日"
    SERIES_OVERFLOW = "系列课时溢出"


class ConflictSeverity(str, Enum):
    """冲突严重程度"""
    HARD = "hard"      # 硬冲突，必须避免
    SOFT = "soft"      # 软冲突，建议避开


# 软冲突类型（只警告不阻止）
SOFT_CONFLICTS = {ConflictType.EXAM_PERIOD, ConflictType.BLACKOUT_DATE}

# 考试日前后禁课天数（默认考试前 3 天 + 考试后 1 天）
EXAM_BUFFER_DAYS_BEFORE = 3
EXAM_BUFFER_DAYS_AFTER = 1

# 平台黑名单日期（节假日等，可以从配置读取或硬编码）
DEFAULT_BLACKOUT_DATES: List[str] = [
    # "2026-10-01",  # 国庆
    # "2026-10-02",
    # "2026-10-03",
]


class Conflict:
    """单个冲突描述"""
    def __init__(
        self,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        message: str,
        related_id: int = None,
        related_date: date = None,
        extra: Dict[str, Any] = None,
    ):
        self.conflict_type = conflict_type
        self.severity = severity
        self.message = message
        self.related_id = related_id
        self.related_date = related_date
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "related_id": self.related_id,
            "related_date": self.related_date.isoformat() if self.related_date else None,
            "extra": self.extra,
        }

    def __repr__(self):
        return f"Conflict({self.conflict_type.value}, {self.severity.value})"


class ConflictDetector:
    """冲突检测器"""

    def __init__(
        self,
        lesson_repository=None,
        blackout_dates: List[str] = None,
    ):
        """
        Args:
            lesson_repository: LessonRepository 实例（可选，懒加载）
            blackout_dates: 平台黑名单日期列表（YYYY-MM-DD）
        """
        self._lesson_repo = lesson_repository
        self.blackout_dates = set(blackout_dates or DEFAULT_BLACKOUT_DATES)

    @property
    def lesson_repo(self):
        if self._lesson_repo is None:
            from db.db_router import DatabaseRouter
            self._lesson_repo = DatabaseRouter().lessons
        return self._lesson_repo

    # ------------------- 单课时冲突检测 -------------------

    def check_single_lesson(
        self,
        teacher_id: int,
        student_id: int,
        scheduled_at: datetime,
        duration_minutes: int,
        exclude_lesson_id: int = None,
        teacher_availability: List[Dict[str, Any]] = None,
        exam_dates: List[date] = None,
    ) -> List[Conflict]:
        """
        检测单个课时的所有冲突

        Returns:
            Conflict 列表（空列表表示无冲突）
        """
        conflicts: List[Conflict] = []
        end_time = scheduled_at + timedelta(minutes=duration_minutes)

        # 1. 老师档期冲突（硬）
        if self.lesson_repo.check_teacher_conflict(
            teacher_id, scheduled_at, end_time, exclude_lesson_id=exclude_lesson_id
        ):
            conflicts.append(Conflict(
                conflict_type=ConflictType.TEACHER_BUSY,
                severity=ConflictSeverity.HARD,
                message=f"老师在该时段已有其他课程",
                related_id=teacher_id,
                related_date=scheduled_at.date(),
            ))

        # 2. 学生档期冲突（硬）
        if self.lesson_repo.check_student_conflict(
            student_id, scheduled_at, end_time, exclude_lesson_id=exclude_lesson_id
        ):
            conflicts.append(Conflict(
                conflict_type=ConflictType.STUDENT_BUSY,
                severity=ConflictSeverity.HARD,
                message=f"学生在该时段已有其他课程",
                related_id=student_id,
                related_date=scheduled_at.date(),
            ))

        # 3. 老师 availability 检查（硬）
        if teacher_availability is not None:
            if not self._check_teacher_availability(
                teacher_availability, scheduled_at, duration_minutes
            ):
                conflicts.append(Conflict(
                    conflict_type=ConflictType.TEACHER_OFF,
                    severity=ConflictSeverity.HARD,
                    message=f"老师在该时段不可用（不在公布的可预约时段内）",
                    related_id=teacher_id,
                    related_date=scheduled_at.date(),
                ))

        # 4. 考试日（软）
        if exam_dates:
            exam_conflict = self._check_exam_period(scheduled_at, exam_dates)
            if exam_conflict:
                conflicts.append(exam_conflict)

        # 5. 黑名单日（软）
        if scheduled_at.date().isoformat() in self.blackout_dates:
            conflicts.append(Conflict(
                conflict_type=ConflictType.BLACKOUT_DATE,
                severity=ConflictSeverity.SOFT,
                message=f"{scheduled_at.date().isoformat()} 是平台黑名单日（节假日等）",
                related_date=scheduled_at.date(),
            ))

        return conflicts

    # ------------------- 系列课时冲突检测 -------------------

    def check_series_lessons(
        self,
        teacher_id: int,
        student_id: int,
        weekly_pattern: List[Dict[str, Any]],
        start_date: date,
        total_weeks: int,
        duration_minutes: int,
        teacher_availability: List[Dict[str, Any]] = None,
        exam_dates: List[date] = None,
        skip_past: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Conflict]]:
        """
        检测系列课的所有课时冲突

        Returns:
            (clean_lessons, all_conflicts)
            clean_lessons: List[{week, day_of_week, scheduled_at, has_conflict, conflicts}]
            all_conflicts: 所有发现的冲突
        """
        lessons_info = []
        all_conflicts: List[Conflict] = []

        current_week_start = start_date
        for week_idx in range(total_weeks):
            for pattern in weekly_pattern:
                    dow = pattern["day_of_week"]
                    time_str = pattern.get("time") or pattern.get("start_time", "19:00")

                    lesson_date = current_week_start + timedelta(days=dow)
                    hour, minute = map(int, time_str.split(":"))
                    scheduled_at = datetime.combine(lesson_date, datetime.min.time()).replace(
                        hour=hour, minute=minute
                    )

                    # 跳过已过去的日期
                    if skip_past and scheduled_at < datetime.now() - timedelta(hours=1):
                        lessons_info.append({
                            "week": week_idx + 1,
                            "scheduled_at": scheduled_at,
                            "skipped": True,
                            "reason": "时间已过",
                            "conflicts": [],
                        })
                        continue

                    conflicts = self.check_single_lesson(
                        teacher_id=teacher_id,
                        student_id=student_id,
                        scheduled_at=scheduled_at,
                        duration_minutes=duration_minutes,
                        teacher_availability=teacher_availability,
                        exam_dates=exam_dates,
                    )

                    lessons_info.append({
                        "week": week_idx + 1,
                        "scheduled_at": scheduled_at,
                        "skipped": False,
                        "conflicts": conflicts,
                    })
                    all_conflicts.extend(conflicts)

            current_week_start += timedelta(weeks=1)

        return lessons_info, all_conflicts

    # ------------------- 套餐溢出检测 -------------------

    def check_series_overflow(
        self,
        package_id: int,
        planned_lessons_count: int,
    ) -> Conflict | None:
        """检查系列课总课时是否超出套餐"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        pkg = db.packages.get_package_by_id(package_id)
        if not pkg:
            return None
        if planned_lessons_count > pkg["total_lessons"]:
            return Conflict(
                conflict_type=ConflictType.SERIES_OVERFLOW,
                severity=ConflictSeverity.HARD,
                message=f"系列课计划 {planned_lessons_count} 课时，超过套餐 {pkg['total_lessons']} 课时",
                related_id=package_id,
                extra={
                    "planned": planned_lessons_count,
                    "package_total": pkg["total_lessons"],
                },
            )
        return None

    # ------------------- 私有方法 -------------------

    def _check_teacher_availability(
        self,
        availability: List[Dict[str, Any]],
        scheduled_at: datetime,
        duration_minutes: int,
    ) -> bool:
        """
        检查给定时刻是否在老师的 availability 内

        availability 格式: [{"day": "monday", "slots": ["19:00-20:00"]}, ...]
        """
        weekday_en_map = {
            0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday",
            4: "friday", 5: "saturday", 6: "sunday",
        }
        weekday = weekday_en_map.get(scheduled_at.weekday())
        if weekday is None:
            return False

        start_str = scheduled_at.strftime("%H:%M")
        end_dt = scheduled_at + timedelta(minutes=duration_minutes)
        end_str = end_dt.strftime("%H:%M")

        for slot in availability:
            if slot.get("day") != weekday:
                continue
            for slot_range in slot.get("slots", []):
                try:
                    s_start, s_end = slot_range.split("-")
                except ValueError:
                    continue
                # 必须完整落在该 slot 内
                if s_start <= start_str and end_str <= s_end:
                    return True
        return False

    def _check_exam_period(
        self,
        scheduled_at: datetime,
        exam_dates: List[date],
    ) -> Conflict | None:
        """检查是否在考试日前后禁课期内"""
        for exam_date in exam_dates:
            # 考试前 3 天 + 考试当天 + 考试后 1 天
            buffer_start = exam_date - timedelta(days=EXAM_BUFFER_DAYS_BEFORE)
            buffer_end = exam_date + timedelta(days=EXAM_BUFFER_DAYS_AFTER)
            if buffer_start <= scheduled_at.date() <= buffer_end:
                return Conflict(
                    conflict_type=ConflictType.EXAM_PERIOD,
                    severity=ConflictSeverity.SOFT,
                    message=f"接近考试日 {exam_date.isoformat()}，建议避开",
                    related_date=exam_date,
                )
        return None