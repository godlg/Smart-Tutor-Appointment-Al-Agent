"""
学生 + 家长 数据访问对象

职责：
1. 家长 CRUD
2. 学生画像 CRUD
3. 学习目标 CRUD
"""

from typing import List, Dict, Any, Optional
from ..base.session_manager import SessionManager
from ..models import Parent, StudentProfile, LearningGoal


class StudentRepository:
    """学生 + 家长 Repository"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ------------------- Parent -------------------

    def add_parent(self, name: str, phone: str = None, budget_max: float = None) -> int:
        with self.session_manager.session_scope() as session:
            p = Parent(name=name, phone=phone, budget_max=budget_max)
            session.add(p)
            session.flush()
            return p.id

    def get_parent_by_id(self, parent_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            p = session.query(Parent).filter(Parent.id == parent_id).first()
            return self._parent_to_dict(p) if p else None

    def get_parent_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            p = session.query(Parent).filter(Parent.phone == phone).first()
            return self._parent_to_dict(p) if p else None

    def get_or_create_default_parent(self) -> Dict[str, Any]:
        """获取或创建默认家长（单用户场景）"""
        with self.session_manager.session_scope() as session:
            p = session.query(Parent).filter(Parent.phone == "default_parent").first()
            if not p:
                p = Parent(name="默认家长", phone="default_parent", budget_max=500.0)
                session.add(p)
                session.flush()
            return self._parent_to_dict(p)

    # ------------------- StudentProfile -------------------

    def add_student(self, parent_id: int, name: str, **kwargs) -> int:
        with self.session_manager.session_scope() as session:
            s = StudentProfile(parent_id=parent_id, name=name, **kwargs)
            session.add(s)
            session.flush()
            return s.id

    def get_student_by_id(self, student_id: int) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            s = session.query(StudentProfile).filter(StudentProfile.id == student_id).first()
            return self._student_to_dict(s) if s else None

    def get_students_by_parent(self, parent_id: int) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            students = session.query(StudentProfile).filter(
                StudentProfile.parent_id == parent_id
            ).all()
            return [self._student_to_dict(s) for s in students]

    def get_all_students(self) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            students = session.query(StudentProfile).all()
            return [self._student_to_dict(s) for s in students]

    def update_student(self, student_id: int, **updates) -> bool:
        with self.session_manager.session_scope() as session:
            s = session.query(StudentProfile).filter(StudentProfile.id == student_id).first()
            if not s:
                return False
            for k, v in updates.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return True

    def update_engagement(self, student_id: int, engagement: float = None,
                          churn_risk: float = None, upsell_signal: str = None) -> bool:
        updates = {}
        if engagement is not None:
            updates["engagement_score"] = engagement
        if churn_risk is not None:
            updates["churn_risk"] = churn_risk
        if upsell_signal is not None:
            updates["upsell_signal"] = upsell_signal
        if updates:
            return self.update_student(student_id, **updates)
        return False

    # ------------------- LearningGoal -------------------

    def add_learning_goal(self, student_id: int, subject: str, description: str,
                          target_score: float = None, deadline=None) -> int:
        with self.session_manager.session_scope() as session:
            g = LearningGoal(
                student_id=student_id, subject=subject,
                description=description, target_score=target_score, deadline=deadline
            )
            session.add(g)
            session.flush()
            return g.id

    def get_student_goals(self, student_id: int) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            goals = session.query(LearningGoal).filter(
                LearningGoal.student_id == student_id
            ).all()
            return [self._goal_to_dict(g) for g in goals]

    # ------------------- 内部方法 -------------------

    def _parent_to_dict(self, p: Parent) -> Dict[str, Any]:
        return {
            "id": p.id, "name": p.name, "phone": p.phone,
            "budget_max": p.budget_max,
            "preferred_contact": p.preferred_contact,
            "created_at": p.created_at,
        }

    def _student_to_dict(self, s: StudentProfile) -> Dict[str, Any]:
        return {
            "id": s.id, "parent_id": s.parent_id, "name": s.name,
            "grade_level": s.grade_level,
            "weak_subjects": s.weak_subjects or [],
            "learning_goals": s.learning_goals or [],
            "preferred_style": s.preferred_style,
            "school": s.school, "notes": s.notes,
            "engagement_score": s.engagement_score,
            "churn_risk": s.churn_risk,
            "upsell_signal": s.upsell_signal,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }

    def _goal_to_dict(self, g: LearningGoal) -> Dict[str, Any]:
        return {
            "id": g.id, "student_id": g.student_id,
            "subject": g.subject, "target_score": g.target_score,
            "deadline": g.deadline, "description": g.description,
            "status": g.status, "created_at": g.created_at,
        }