"""
老师数据访问对象

职责：
1. 老师信息的 CRUD
2. 按学科/年级/教学风格筛选
3. 可用时段查询
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import or_, and_
from ..base.interfaces import BaseTeacherRepository
from ..base.session_manager import SessionManager
from ..models import Teacher


class TeacherRepository(BaseTeacherRepository):
    """老师 Repository"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ------------------- CRUD -------------------

    def add_teacher(self, **kwargs) -> int:
        """添加老师（任意字段）"""
        with self.session_manager.session_scope() as session:
            teacher = Teacher(**kwargs)
            session.add(teacher)
            session.flush()
            return teacher.id

    def get_teacher_by_id(self, teacher_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            t = session.query(Teacher).filter(Teacher.id == teacher_id).first()
            return self._to_dict(t) if t else None

    def get_teacher_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            t = session.query(Teacher).filter(Teacher.name == name).first()
            return self._to_dict(t) if t else None

    def get_all_teachers(self, only_active: bool = True) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            q = session.query(Teacher)
            if only_active:
                q = q.filter(Teacher.is_active == True)
            return [self._to_dict(t) for t in q.all()]

    def update_teacher(self, teacher_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            t = session.query(Teacher).filter(Teacher.id == teacher_id).first()
            if not t:
                return False
            for k, v in updates.items():
                if hasattr(t, k):
                    setattr(t, k, v)
            return True

    def delete_teacher(self, teacher_id: int) -> bool:
        with self.session_manager.session_scope() as session:
            t = session.query(Teacher).filter(Teacher.id == teacher_id).first()
            if not t:
                return False
            t.is_active = False  # 软删除
            return True

    # ------------------- 查询筛选 -------------------

    def find_teachers_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """按学科筛选（subjects 是 JSON list）"""
        all_t = self.get_all_teachers()
        return [t for t in all_t if subject in (t.get("subjects") or [])]

    def find_teachers_by_grade(self, grade_level: str) -> List[Dict[str, Any]]:
        all_t = self.get_all_teachers()
        return [t for t in all_t if grade_level in (t.get("grade_levels") or [])]

    def find_teachers_by_subject_and_grade(
        self, subject: str, grade_level: str
    ) -> List[Dict[str, Any]]:
        all_t = self.get_all_teachers()
        return [
            t for t in all_t
            if subject in (t.get("subjects") or [])
            and grade_level in (t.get("grade_levels") or [])
        ]

    def get_available_teachers(self) -> List[Dict[str, Any]]:
        """提供试讲的老师"""
        all_t = self.get_all_teachers()
        return [t for t in all_t if t.get("trial_available", True)]

    def get_all_subjects(self) -> List[str]:
        """所有老师涉及的学科"""
        all_t = self.get_all_teachers()
        s = set()
        for t in all_t:
            for sub in (t.get("subjects") or []):
                s.add(sub)
        return sorted(s)

    def get_all_grade_levels(self) -> List[str]:
        all_t = self.get_all_teachers()
        s = set()
        for t in all_t:
            for g in (t.get("grade_levels") or []):
                s.add(g)
        return sorted(s)

    # ------------------- 可用时段 -------------------

    def get_teacher_availability(self, teacher_id: int) -> List[Dict[str, Any]]:
        t = self.get_teacher_by_id(teacher_id)
        return t.get("availability") or [] if t else []

    def is_teacher_available_at(self, teacher_id: int, weekday: str, time_slot: str) -> bool:
        """检查老师是否在某 weekday + time_slot 可用"""
        t = self.get_teacher_by_id(teacher_id)
        if not t:
            return False
        for slot in (t.get("availability") or []):
            if slot.get("day") == weekday and time_slot in slot.get("slots", []):
                return True
        return False

    # ------------------- 内部方法 -------------------

    def _to_dict(self, t: Teacher) -> Dict[str, Any]:
        return {
            "id": t.id,
            "name": t.name,
            "avatar_url": t.avatar_url,
            "gender": t.gender,
            "subjects": t.subjects or [],
            "grade_levels": t.grade_levels or [],
            "teaching_style": t.teaching_style,
            "education_bg": t.education_bg,
            "years_experience": t.years_experience,
            "hourly_rate": t.hourly_rate,
            "rating": t.rating,
            "availability": t.availability or [],
            "is_active": t.is_active,
            "trial_available": t.trial_available,
            "bio": t.bio,
            "highlights": t.highlights or [],
            "created_at": t.created_at,
        }