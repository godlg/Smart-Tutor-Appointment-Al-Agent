"""
课时 + 系列 + 试讲反馈 数据访问对象
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta, timezone
from sqlalchemy import and_, or_
from ..base.session_manager import SessionManager
from ..models import Lesson, LessonSeries, TrialFeedback

# 库内 scheduled_at 存的是北京时间 naive（无 tzinfo）。这里直接构造时区，不 import config，
# 保持 db 层不依赖业务配置。
_BEIJING_TZ = timezone(timedelta(hours=8))


def _to_db_naive(value):
    """把入参归一成与库内一致的北京时间 naive，用于和 DB 行做比较。

    aware → 转北京时间后去掉 tzinfo；naive（或 None、date）原样返回，零行为变化。
    只作用于查询入参，不改 DB 行的值。
    """
    if getattr(value, "tzinfo", None) is None:
        return value
    return value.astimezone(_BEIJING_TZ).replace(tzinfo=None)


class LessonRepository:
    """Lesson 系列 Repository"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ------------------- Lesson CRUD -------------------

    def add_lesson(
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
    ) -> int:
        with self.session_manager.session_scope() as session:
            lesson = Lesson(
                teacher_id=teacher_id,
                student_id=student_id,
                scheduled_at=scheduled_at,
                lesson_type=lesson_type,
                subject=subject,
                duration_minutes=duration_minutes,
                status=status,
                series_id=series_id,
                series_index=series_index,
                package_id=package_id,
                price=price,
            )
            session.add(lesson)
            session.flush()
            return lesson.id

    def get_lesson_by_id(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            l = session.query(Lesson).filter(Lesson.id == lesson_id).first()
            return self._lesson_to_dict(l) if l else None

    def get_lessons_by_teacher(
        self, teacher_id: int, status: str = None,
        start_date: datetime = None, end_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        start_date = _to_db_naive(start_date)
        end_date = _to_db_naive(end_date)
        with self.session_manager.session_scope() as session:
            q = session.query(Lesson).filter(Lesson.teacher_id == teacher_id)
            if status:
                q = q.filter(Lesson.status == status)
            if start_date:
                q = q.filter(Lesson.scheduled_at >= start_date)
            if end_date:
                q = q.filter(Lesson.scheduled_at <= end_date)
            return [self._lesson_to_dict(l) for l in q.order_by(Lesson.scheduled_at).all()]

    def get_lessons_by_student(
        self, student_id: int, status: str = None,
        lesson_type: str = None,
    ) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            q = session.query(Lesson).filter(Lesson.student_id == student_id)
            if status:
                q = q.filter(Lesson.status == status)
            if lesson_type:
                q = q.filter(Lesson.lesson_type == lesson_type)
            return [self._lesson_to_dict(l) for l in q.order_by(Lesson.scheduled_at).all()]

    def get_lessons_by_series(self, series_id: int) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            lessons = session.query(Lesson).filter(
                Lesson.series_id == series_id
            ).order_by(Lesson.series_index).all()
            return [self._lesson_to_dict(l) for l in lessons]

    def get_pending_trial_lessons(self, parent_id: int = None) -> List[Dict[str, Any]]:
        """获取待试讲的课时"""
        return self.get_lessons_by_type_status("trial", "pending")

    def get_lessons_by_type_status(self, lesson_type: str, status: str) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            lessons = session.query(Lesson).filter(
                and_(Lesson.lesson_type == lesson_type, Lesson.status == status)
            ).order_by(Lesson.scheduled_at).all()
            return [self._lesson_to_dict(l) for l in lessons]

    def update_lesson(self, lesson_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            l = session.query(Lesson).filter(Lesson.id == lesson_id).first()
            if not l:
                return False
            for k, v in updates.items():
                if hasattr(l, k):
                    setattr(l, k, v)
            return True

    def cancel_lesson(self, lesson_id: int) -> bool:
        return self.update_lesson(lesson_id, status="cancelled")

    def complete_lesson(self, lesson_id: int) -> bool:
        return self.update_lesson(lesson_id, status="completed")

    def reschedule_lesson(self, lesson_id: int, new_time: datetime) -> bool:
        return self.update_lesson(lesson_id, scheduled_at=new_time, status="confirmed")

    def count_lessons_by_type(self, student_id: int, lesson_type: str,
                              status: str = None) -> int:
        with self.session_manager.session_scope() as session:
            q = session.query(Lesson).filter(
                Lesson.student_id == student_id,
                Lesson.lesson_type == lesson_type,
            )
            if status:
                q = q.filter(Lesson.status == status)
            return q.count()

    # ------------------- 冲突检测 -------------------

    def check_teacher_conflict(
        self, teacher_id: int, start_time: datetime, end_time: datetime,
        exclude_lesson_id: int = None,
    ) -> bool:
        """检查老师在该时段是否已被预约（返回 True 表示有冲突）"""
        start_time = _to_db_naive(start_time)
        end_time = _to_db_naive(end_time)
        with self.session_manager.session_scope() as session:
            q = session.query(Lesson).filter(
                Lesson.teacher_id == teacher_id,
                Lesson.status.in_(["pending", "confirmed"]),
                Lesson.scheduled_at < end_time,
            )
            if exclude_lesson_id:
                q = q.filter(Lesson.id != exclude_lesson_id)
            for l in q.all():
                l_end = l.scheduled_at.replace()  # 简化：用 scheduled_at 近似
                # 实际应该 l.scheduled_at + duration_minutes
                from datetime import timedelta
                l_end = l.scheduled_at + timedelta(minutes=l.duration_minutes)
                if l.scheduled_at < end_time and l_end > start_time:
                    return True
            return False

    def check_student_conflict(
        self, student_id: int, start_time: datetime, end_time: datetime,
        exclude_lesson_id: int = None,
    ) -> bool:
        start_time = _to_db_naive(start_time)
        end_time = _to_db_naive(end_time)
        with self.session_manager.session_scope() as session:
            q = session.query(Lesson).filter(
                Lesson.student_id == student_id,
                Lesson.status.in_(["pending", "confirmed"]),
            )
            if exclude_lesson_id:
                q = q.filter(Lesson.id != exclude_lesson_id)
            from datetime import timedelta
            for l in q.all():
                l_end = l.scheduled_at + timedelta(minutes=l.duration_minutes)
                if l.scheduled_at < end_time and l_end > start_time:
                    return True
            return False

    # ------------------- LessonSeries CRUD -------------------

    def add_series(
        self, teacher_id: int, student_id: int,
        start_date: date, end_date: date,
        weekly_pattern: list, total_lessons: int,
        package_id: int = None,
    ) -> int:
        with self.session_manager.session_scope() as session:
            s = LessonSeries(
                teacher_id=teacher_id, student_id=student_id,
                start_date=start_date, end_date=end_date,
                weekly_pattern=weekly_pattern,
                total_lessons=total_lessons,
                package_id=package_id,
            )
            session.add(s)
            session.flush()
            return s.id

    def get_series_by_id(self, series_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            s = session.query(LessonSeries).filter(LessonSeries.id == series_id).first()
            return self._series_to_dict(s) if s else None

    def get_series_by_student(self, student_id: int) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            series_list = session.query(LessonSeries).filter(
                LessonSeries.student_id == student_id
            ).all()
            return [self._series_to_dict(s) for s in series_list]

    def update_series(self, series_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            s = session.query(LessonSeries).filter(LessonSeries.id == series_id).first()
            if not s:
                return False
            for k, v in updates.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return True

    def increment_series_completed(self, series_id: int) -> bool:
        with self.session_manager.session_scope() as session:
            s = session.query(LessonSeries).filter(LessonSeries.id == series_id).first()
            if not s:
                return False
            s.completed_lessons = (s.completed_lessons or 0) + 1
            return True

    # ------------------- TrialFeedback CRUD -------------------

    def add_trial_feedback(
        self, lesson_id: int, parent_id: int,
        rating: int, would_continue: bool,
        feedback_text: str = None, recommended_subjects: list = None,
    ) -> int:
        with self.session_manager.session_scope() as session:
            f = TrialFeedback(
                lesson_id=lesson_id, parent_id=parent_id,
                rating=rating, would_continue=would_continue,
                feedback_text=feedback_text,
                recommended_subjects=recommended_subjects,
            )
            session.add(f)
            session.flush()
            return f.id

    def get_feedback_by_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            f = session.query(TrialFeedback).filter(
                TrialFeedback.lesson_id == lesson_id
            ).first()
            return self._feedback_to_dict(f) if f else None

    # ------------------- 内部方法 -------------------

    def _lesson_to_dict(self, l: Lesson) -> Dict[str, Any]:
        return {
            "id": l.id,
            "teacher_id": l.teacher_id,
            "student_id": l.student_id,
            "package_id": l.package_id,
            "series_id": l.series_id,
            "series_index": l.series_index,
            "lesson_type": l.lesson_type,
            "subject": l.subject,
            "scheduled_at": l.scheduled_at,
            "duration_minutes": l.duration_minutes,
            "status": l.status,
            "progress_record_id": l.progress_record_id,
            "parent_feedback": l.parent_feedback,
            "price": l.price,
            "created_at": l.created_at,
            "updated_at": l.updated_at,
        }

    def _series_to_dict(self, s: LessonSeries) -> Dict[str, Any]:
        return {
            "id": s.id,
            "teacher_id": s.teacher_id,
            "student_id": s.student_id,
            "package_id": s.package_id,
            "start_date": s.start_date,
            "end_date": s.end_date,
            "weekly_pattern": s.weekly_pattern or [],
            "total_lessons": s.total_lessons,
            "completed_lessons": s.completed_lessons,
            "cancelled_lessons": s.cancelled_lessons,
            "status": s.status,
            "created_at": s.created_at,
        }

    def _feedback_to_dict(self, f: TrialFeedback) -> Dict[str, Any]:
        return {
            "id": f.id,
            "lesson_id": f.lesson_id,
            "parent_id": f.parent_id,
            "rating": f.rating,
            "would_continue": f.would_continue,
            "feedback_text": f.feedback_text,
            "recommended_subjects": f.recommended_subjects or [],
            "created_at": f.created_at,
        }