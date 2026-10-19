"""
课时服务层

职责：
1. 课时 CRUD
2. 冲突检测
3. 系列生成（周固定）
4. 调课/退课
"""

import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class LessonService:
    """课时业务服务"""

    def __init__(self):
        self.db = DatabaseRouter()

    @property
    def lesson_repo(self):
        return self.db.lessons

    @property
    def teacher_repo(self):
        return self.db.teachers

    # ------------------- 单次课时 -------------------

    def create_lesson(
        self,
        teacher_id: int,
        student_id: int,
        scheduled_at: datetime,
        lesson_type: str = "regular",
        subject: str = None,
        duration_minutes: int = 60,
        status: str = "pending",
        series_id: int = None,
        series_index: int = None,
        package_id: int = None,
        price: float = None,
    ) -> Tuple[bool, Any]:
        """
        创建课时（自动做冲突检测）

        Returns:
            (success, lesson_id_or_error_message)
        """
        # 冲突检测
        end_time = scheduled_at + timedelta(minutes=duration_minutes)

        if self.lesson_repo.check_teacher_conflict(teacher_id, scheduled_at, end_time):
            return False, "老师在该时段已有课程，请选择其他时间"

        if self.lesson_repo.check_student_conflict(student_id, scheduled_at, end_time):
            return False, "学生在该时段已有其他课程，请选择其他时间"

        lesson_id = self.lesson_repo.add_lesson(
            teacher_id=teacher_id, student_id=student_id,
            scheduled_at=scheduled_at, lesson_type=lesson_type,
            subject=subject, duration_minutes=duration_minutes,
            status=status, series_id=series_id,
            series_index=series_index, package_id=package_id, price=price,
        )
        logger.info(f"创建课时成功: teacher={teacher_id}, student={student_id}, time={scheduled_at}, type={lesson_type}")
        return True, lesson_id

    def get_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        return self.lesson_repo.get_lesson_by_id(lesson_id)

    def get_lessons_by_teacher(self, teacher_id: int, status: str = None,
                               start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        return self.lesson_repo.get_lessons_by_teacher(teacher_id, status, start_date, end_date)

    def get_lessons_by_student(self, student_id: int, status: str = None,
                               lesson_type: str = None) -> List[Dict[str, Any]]:
        return self.lesson_repo.get_lessons_by_student(student_id, status, lesson_type)

    def get_lessons_by_series(self, series_id: int) -> List[Dict[str, Any]]:
        return self.lesson_repo.get_lessons_by_series(series_id)

    def get_pending_trials(self) -> List[Dict[str, Any]]:
        return self.lesson_repo.get_pending_trial_lessons()

    def cancel_lesson(self, lesson_id: int) -> bool:
        return self.lesson_repo.cancel_lesson(lesson_id)

    def complete_lesson(self, lesson_id: int) -> bool:
        success = self.lesson_repo.complete_lesson(lesson_id)
        if success:
            lesson = self.lesson_repo.get_lesson_by_id(lesson_id)
            if lesson and lesson.get("series_id"):
                self.lesson_repo.increment_series_completed(lesson["series_id"])
        return success

    def reschedule_lesson(self, lesson_id: int, new_time: datetime) -> Tuple[bool, str]:
        """调课（单次）"""
        lesson = self.lesson_repo.get_lesson_by_id(lesson_id)
        if not lesson:
            return False, "课时不存在"
        if lesson["status"] in ["completed", "cancelled"]:
            return False, f"课时状态为 {lesson['status']}，不可调课"

        end_time = new_time + timedelta(minutes=lesson["duration_minutes"])

        if self.lesson_repo.check_teacher_conflict(
            lesson["teacher_id"], new_time, end_time, exclude_lesson_id=lesson_id
        ):
            return False, "新时间老师档期冲突"

        if self.lesson_repo.check_student_conflict(
            lesson["student_id"], new_time, end_time, exclude_lesson_id=lesson_id
        ):
            return False, "新时间学生已有其他课程"

        success = self.lesson_repo.reschedule_lesson(lesson_id, new_time)
        return (True, "调课成功") if success else (False, "调课失败")

    def update_lesson_status(self, lesson_id: int, status: str) -> bool:
        return self.lesson_repo.update_lesson(lesson_id, status=status)

    def update_lesson(self, lesson_id: int, **kwargs) -> bool:
        return self.lesson_repo.update_lesson(lesson_id, **kwargs)

    # ------------------- 系列课（周固定）-------------------

    def create_lesson_series(
        self,
        teacher_id: int,
        student_id: int,
        weekly_pattern: List[Dict[str, Any]],
        start_date: date,
        total_weeks: int,
        subject: str = None,
        duration_minutes: int = 60,
        package_id: int = None,
        skip_conflicts: bool = True,
    ) -> Tuple[bool, Any]:
        """
        创建系列课（周固定排课）

        Phase 2 重构：使用 agents.lesson.ScheduleGenerator 实现。
        保留此函数是为了向后兼容原有调用方。

        Args:
            weekly_pattern: [{"day_of_week": 3, "time": "19:00"}, ...]
                           day_of_week: 0=周一, 6=周日
            start_date: 起始日期
            total_weeks: 总周数

        Returns:
            (success, {"series_id": int, "lessons_created": int, "skipped": list})
        """
        from agents.lesson import ScheduleGenerator, ConflictDetector

        # 1. 准备参数（使用新的生成器）
        teacher_avail = None
        teacher = self.teacher_repo.get_teacher_by_id(teacher_id)
        if teacher:
            teacher_avail = teacher.get("availability")

        generator = ScheduleGenerator(ConflictDetector())
        normalized_pattern = generator.parse_weekly_pattern(weekly_pattern)

        gen_result = generator.generate_series_lessons(
            teacher_id=teacher_id,
            student_id=student_id,
            start_date=start_date,
            total_weeks=total_weeks,
            weekly_pattern=normalized_pattern,
            duration_minutes=duration_minutes,
            subject=subject,
            package_id=package_id,
            teacher_availability=teacher_avail,
            skip_hard_conflicts=skip_conflicts,
        )

        if gen_result.get("error"):
            return False, {"error": gen_result["error"], "conflicts": gen_result["conflicts"]}

        # 收集 skipped 信息（合并 hard 冲突 + 时间过期）
        skipped = []
        for conflict in gen_result["conflicts"]:
            if conflict.get("severity") == "hard":
                skipped.append({
                    "reason": conflict.get("message"),
                    "conflict_type": conflict.get("conflict_type"),
                })
        skipped.extend([
            {"reason": w} for w in gen_result.get("warnings", [])
        ])

        # 所有课时都被跳过时，不创建空系列记录
        if not gen_result["lessons"]:
            logger.warning(
                f"创建系列课失败: 全部 {len(skipped)} 个课时被跳过，不创建空系列 "
                f"(teacher={teacher_id}, student={student_id})"
            )
            return False, {"error": "系列课全部课时被跳过", "skipped": skipped}

        # 2. 创建系列记录
        end_date = start_date + timedelta(weeks=total_weeks) - timedelta(days=1)
        total_lessons = len(gen_result["lessons"])

        series_id = self.lesson_repo.add_series(
            teacher_id=teacher_id, student_id=student_id,
            start_date=start_date, end_date=end_date,
            weekly_pattern=normalized_pattern,
            total_lessons=total_lessons,
            package_id=package_id,
        )

        # 3. 批量创建 lesson
        lessons_created = 0

        for lesson_data in gen_result["lessons"]:
            self.lesson_repo.add_lesson(
                teacher_id=lesson_data["teacher_id"],
                student_id=lesson_data["student_id"],
                scheduled_at=lesson_data["scheduled_at"],
                lesson_type="series",
                subject=lesson_data["subject"],
                duration_minutes=lesson_data["duration_minutes"],
                status="confirmed",
                series_id=series_id,
                series_index=lesson_data["series_index"],
                package_id=lesson_data["package_id"],
            )
            lessons_created += 1

        logger.info(
            f"创建系列课: series_id={series_id}, 共生成 {lessons_created} 个课时，"
            f"跳过 {len(skipped)} 个"
        )

        return True, {
            "series_id": series_id,
            "lessons_created": lessons_created,
            "skipped": skipped,
        }

    def get_series(self, series_id: int) -> Optional[Dict[str, Any]]:
        return self.lesson_repo.get_series_by_id(series_id)

    def get_series_lessons(self, series_id: int) -> List[Dict[str, Any]]:
        return self.lesson_repo.get_lessons_by_series(series_id)

    def pause_series(self, series_id: int) -> bool:
        return self.lesson_repo.update_series(series_id, status="paused")

    def resume_series(self, series_id: int) -> bool:
        return self.lesson_repo.update_series(series_id, status="active")

    # ------------------- 统计 -------------------

    def count_student_lessons(self, student_id: int, lesson_type: str = None,
                              status: str = None) -> int:
        if lesson_type:
            return self.lesson_repo.count_lessons_by_type(student_id, lesson_type, status)
        # 全部课时
        lessons = self.lesson_repo.get_lessons_by_student(student_id, status)
        return len(lessons)

    def get_student_upcoming_lessons(self, student_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """获取学生即将到来的课时"""
        all_lessons = self.lesson_repo.get_lessons_by_student(
            student_id, status="confirmed"
        )
        upcoming = [
            l for l in all_lessons
            if l["scheduled_at"] and l["scheduled_at"] > datetime.now()
        ]
        return upcoming[:limit]

    # ------------------- 试讲反馈 -------------------

    def add_trial_feedback(
        self, lesson_id: int, parent_id: int,
        rating: int, would_continue: bool,
        feedback_text: str = None, recommended_subjects: list = None,
    ) -> int:
        return self.lesson_repo.add_trial_feedback(
            lesson_id=lesson_id, parent_id=parent_id,
            rating=rating, would_continue=would_continue,
            feedback_text=feedback_text,
            recommended_subjects=recommended_subjects,
        )

    def get_trial_feedback(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        return self.lesson_repo.get_feedback_by_lesson(lesson_id)