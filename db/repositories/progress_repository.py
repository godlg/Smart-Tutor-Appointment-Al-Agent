"""
课后进度记录 Repository
"""

from typing import List, Dict, Any, Optional
from ..base.session_manager import SessionManager
from ..models import ProgressRecord


class ProgressRepository:
    """ProgressRecord 数据访问"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def add_progress(self, **kwargs) -> int:
        with self.session_manager.session_scope() as session:
            p = ProgressRecord(**kwargs)
            session.add(p)
            session.flush()
            return p.id

    def get_progress_by_id(self, progress_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            p = session.query(ProgressRecord).filter(ProgressRecord.id == progress_id).first()
            return self._to_dict(p) if p else None

    def get_progress_by_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            p = session.query(ProgressRecord).filter(
                ProgressRecord.lesson_id == lesson_id
            ).first()
            return self._to_dict(p) if p else None

    def get_progress_by_student(self, student_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            records = session.query(ProgressRecord).filter(
                ProgressRecord.student_id == student_id
            ).order_by(ProgressRecord.created_at.desc()).limit(limit).all()
            return [self._to_dict(p) for p in records]

    def get_progress_by_subject(self, student_id: int, subject: str) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            records = session.query(ProgressRecord).filter(
                ProgressRecord.student_id == student_id,
                ProgressRecord.weak_points.isnot(None),
            ).all()
            # weak_points 是 JSON list，过滤包含 subject 的记录
            return [self._to_dict(r) for r in records if subject in str(r.weak_points or "")]

    def update_progress(self, progress_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            p = session.query(ProgressRecord).filter(ProgressRecord.id == progress_id).first()
            if not p:
                return False
            for k, v in updates.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            return True

    def _to_dict(self, p: ProgressRecord) -> Dict[str, Any]:
        return {
            "id": p.id,
            "lesson_id": p.lesson_id,
            "student_id": p.student_id,
            "teacher_id": p.teacher_id,
            "content_covered": p.content_covered,
            "mastery_score": p.mastery_score,
            "homework": p.homework,
            "weak_points": p.weak_points or [],
            "ai_summary": p.ai_summary,
            "teacher_comment": p.teacher_comment,
            "parent_comment": p.parent_comment,
            "created_at": p.created_at,
        }