"""
调课处理器（带级联处理）

按 TUTOR_DESIGN.md §9.3 实现：
- 单次课调课：直接修改 Lesson.scheduled_at
- 系列课调课：
  ├─ 只调一次：仅改当前 Lesson，series 不变
  └─ 调整后续所有周：级联更新 LessonSeries 之后所有 Lesson
- 冲突检测
- 通知老师
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

from .conflict_detector import ConflictDetector, Conflict, ConflictSeverity

logger = logging.getLogger(__name__)


class RescheduleMode(str):
    """调课模式"""
    ONLY_THIS = "only_this"               # 只调这一次
    THIS_AND_FOLLOWING = "this_and_following"  # 这次及之后所有


class RescheduleHandler:
    """调课处理器"""

    def __init__(self, conflict_detector: ConflictDetector = None):
        self.conflict_detector = conflict_detector or ConflictDetector()

    def reschedule(
        self,
        lesson_id: int,
        new_time: datetime,
        mode: str = RescheduleMode.ONLY_THIS,
    ) -> Tuple[bool, str]:
        """
        调课主入口

        Returns:
            (success, message)
        """
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        lesson = db.lessons.get_lesson_by_id(lesson_id)
        if not lesson:
            return False, "课时不存在"

        if lesson["status"] in ("completed", "cancelled"):
            return False, f"课时状态为 {lesson['status']}，不可调课"

        # 1. 冲突检测
        teacher_avail = None
        teacher = db.teachers.get_teacher_by_id(lesson["teacher_id"])
        if teacher:
            teacher_avail = teacher.get("availability")

        conflicts = self.conflict_detector.check_single_lesson(
            teacher_id=lesson["teacher_id"],
            student_id=lesson["student_id"],
            scheduled_at=new_time,
            duration_minutes=lesson["duration_minutes"],
            exclude_lesson_id=lesson_id,
            teacher_availability=teacher_avail,
        )
        hard = [c for c in conflicts if c.severity == ConflictSeverity.HARD]
        if hard:
            return False, f"新时间冲突：{hard[0].message}"

        # 2. 执行调课
        if mode == RescheduleMode.ONLY_THIS or not lesson.get("series_id"):
            return self._reschedule_single(db, lesson, new_time)
        else:
            return self._reschedule_cascade(db, lesson, new_time)

    def _reschedule_single(
        self,
        db,
        lesson: Dict[str, Any],
        new_time: datetime,
    ) -> Tuple[bool, str]:
        """只调这一次（单次课或系列中的某一节）"""
        success = db.lessons.reschedule_lesson(lesson["id"], new_time)
        if success:
            logger.info(f"调课成功: lesson_id={lesson['id']}, new_time={new_time}")
            return True, "调课成功"
        return False, "数据库更新失败"

    def _reschedule_cascade(
        self,
        db,
        lesson: Dict[str, Any],
        new_time: datetime,
    ) -> Tuple[bool, str]:
        """
        级联调整：修改当前 lesson 及之后所有 series 课

        规则：
        - 当前 lesson 移到 new_time
        - 后续所有 lesson 整体平移（保留每周规律）
        """
        series_id = lesson["series_id"]
        all_lessons = db.lessons.get_lessons_by_series(series_id)
        # 按 scheduled_at 升序
        all_lessons.sort(key=lambda l: l["scheduled_at"])

        # 找到当前 lesson 在系列中的位置
        target_index = None
        for i, l in enumerate(all_lessons):
            if l["id"] == lesson["id"]:
                target_index = i
                break
        if target_index is None:
            return False, "系列中找不到该课时"

        original_dt = lesson["scheduled_at"]
        delta = new_time - original_dt

        updated_count = 0
        skipped_count = 0
        skipped_lessons = []

        teacher = db.teachers.get_teacher_by_id(lesson["teacher_id"])
        teacher_avail = teacher.get("availability") if teacher else None

        for i in range(target_index, len(all_lessons)):
            l = all_lessons[i]
            new_dt = l["scheduled_at"] + delta

            # 已完成或已取消的课不调整
            if l["status"] in ("completed", "cancelled"):
                skipped_count += 1
                continue

            # 冲突检查（排除自身）
            conflicts = self.conflict_detector.check_single_lesson(
                teacher_id=l["teacher_id"],
                student_id=l["student_id"],
                scheduled_at=new_dt,
                duration_minutes=l["duration_minutes"],
                exclude_lesson_id=l["id"],
                teacher_availability=teacher_avail,
            )
            hard = [c for c in conflicts if c.severity == ConflictSeverity.HARD]
            if hard:
                skipped_lessons.append({
                    "lesson_id": l["id"],
                    "new_time": new_dt.isoformat(),
                    "reason": hard[0].message,
                })
                skipped_count += 1
                continue

            db.lessons.reschedule_lesson(l["id"], new_dt)
            updated_count += 1

        if updated_count == 0:
            return False, f"级联调课全部失败：{skipped_lessons}"

        msg = f"级联调课成功：更新 {updated_count} 节，跳过 {skipped_count} 节"
        if skipped_lessons:
            msg += f"；冲突明细：{skipped_lessons}"
        logger.info(msg)
        return True, msg

    def preview_cascade(
        self,
        lesson_id: int,
        new_time: datetime,
    ) -> Dict[str, Any]:
        """
        预览级联调课结果（不实际修改）

        用于前端展示给用户确认
        """
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        lesson = db.lessons.get_lesson_by_id(lesson_id)
        if not lesson:
            return {"error": "课时不存在"}

        if not lesson.get("series_id"):
            return {
                "is_series": False,
                "affected_lessons": 1,
                "conflicts": [],
            }

        all_lessons = db.lessons.get_lessons_by_series(lesson["series_id"])
        all_lessons.sort(key=lambda l: l["scheduled_at"])
        target_index = next(
            (i for i, l in enumerate(all_lessons) if l["id"] == lesson_id),
            None,
        )
        if target_index is None:
            return {"error": "系列中找不到该课时"}

        original_dt = lesson["scheduled_at"]
        delta = new_time - original_dt

        teacher = db.teachers.get_teacher_by_id(lesson["teacher_id"])
        teacher_avail = teacher.get("availability") if teacher else None

        affected = []
        for i in range(target_index, len(all_lessons)):
            l = all_lessons[i]
            new_dt = l["scheduled_at"] + delta
            conflicts = self.conflict_detector.check_single_lesson(
                teacher_id=l["teacher_id"],
                student_id=l["student_id"],
                scheduled_at=new_dt,
                duration_minutes=l["duration_minutes"],
                exclude_lesson_id=l["id"],
                teacher_availability=teacher_avail,
            )
            affected.append({
                "lesson_id": l["id"],
                "original_time": l["scheduled_at"].isoformat(),
                "new_time": new_dt.isoformat(),
                "status": l["status"],
                "conflicts": [c.to_dict() for c in conflicts],
            })

        return {
            "is_series": True,
            "series_id": lesson["series_id"],
            "affected_lessons": len(affected),
            "details": affected,
        }