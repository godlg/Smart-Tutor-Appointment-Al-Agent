"""
课时套餐 Repository
"""

from typing import List, Dict, Any, Optional
from ..base.session_manager import SessionManager
from ..models import CoursePackage


class PackageRepository:
    """CoursePackage 数据访问"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def add_package(self, **kwargs) -> int:
        with self.session_manager.session_scope() as session:
            p = CoursePackage(**kwargs)
            session.add(p)
            session.flush()
            return p.id

    def get_package_by_id(self, package_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            p = session.query(CoursePackage).filter(CoursePackage.id == package_id).first()
            return self._to_dict(p) if p else None

    def get_all_packages(
        self, only_active: bool = True,
        subject: str = None, grade_level: str = None,
        teacher_id: int = None,
    ) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            q = session.query(CoursePackage)
            if only_active:
                q = q.filter(CoursePackage.is_active == True)
            if subject:
                q = q.filter(CoursePackage.subject == subject)
            if grade_level:
                q = q.filter(CoursePackage.grade_level == grade_level)
            if teacher_id:
                q = q.filter(CoursePackage.teacher_id == teacher_id)
            return [self._to_dict(p) for p in q.all()]

    def update_package(self, package_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            p = session.query(CoursePackage).filter(CoursePackage.id == package_id).first()
            if not p:
                return False
            for k, v in updates.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            return True

    def delete_package(self, package_id: int) -> bool:
        with self.session_manager.session_scope() as session:
            p = session.query(CoursePackage).filter(CoursePackage.id == package_id).first()
            if not p:
                return False
            p.is_active = False
            return True

    def _to_dict(self, p: CoursePackage) -> Dict[str, Any]:
        return {
            "id": p.id,
            "name": p.name,
            "teacher_id": p.teacher_id,
            "subject": p.subject,
            "grade_level": p.grade_level,
            "total_lessons": p.total_lessons,
            "duration_minutes": p.duration_minutes,
            "price_total": p.price_total,
            "price_per_lesson": p.price_per_lesson,
            "weekly_schedule": p.weekly_schedule or [],
            "validity_days": p.validity_days,
            "description": p.description,
            "is_active": p.is_active,
            "created_at": p.created_at,
        }