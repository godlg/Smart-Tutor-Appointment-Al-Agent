"""
学生画像服务层
"""

import logging
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class StudentProfileService:
    """学生 + 家长 业务服务"""

    def __init__(self):
        self.db = DatabaseRouter()

    @property
    def student_repo(self):
        return self.db.students

    # ------------------- 家长 -------------------

    def get_or_create_default_parent(self) -> Dict[str, Any]:
        return self.student_repo.get_or_create_default_parent()

    def get_parent(self, parent_id: int) -> Optional[Dict[str, Any]]:
        return self.student_repo.get_parent_by_id(parent_id)

    def add_parent(self, name: str, phone: str = None, budget_max: float = None) -> int:
        return self.student_repo.add_parent(name, phone, budget_max)

    # ------------------- 学生 -------------------

    def add_student(self, parent_id: int, name: str, **kwargs) -> int:
        return self.student_repo.add_student(parent_id, name, **kwargs)

    def get_student(self, student_id: int) -> Optional[Dict[str, Any]]:
        return self.student_repo.get_student_by_id(student_id)

    def get_students_by_parent(self, parent_id: int) -> List[Dict[str, Any]]:
        return self.student_repo.get_students_by_parent(parent_id)

    def get_all_students(self) -> List[Dict[str, Any]]:
        return self.student_repo.get_all_students()

    def update_student(self, student_id: int, **kwargs) -> bool:
        return self.student_repo.update_student(student_id, **kwargs)

    def update_engagement(self, student_id: int, **kwargs) -> bool:
        return self.student_repo.update_engagement(student_id, **kwargs)

    # ------------------- 学习目标 -------------------

    def add_learning_goal(self, student_id: int, subject: str, description: str,
                          target_score: float = None, deadline=None) -> int:
        return self.student_repo.add_learning_goal(
            student_id, subject, description, target_score, deadline
        )

    def get_student_goals(self, student_id: int) -> List[Dict[str, Any]]:
        return self.student_repo.get_student_goals(student_id)

    # ------------------- 初始化默认学生 -------------------

    def initialize_default_student(self) -> Dict[str, Any]:
        """为默认家长创建一个默认学生"""
        parent = self.get_or_create_default_parent()
        students = self.get_students_by_parent(parent["id"])
        if students:
            return students[0]

        student_id = self.add_student(
            parent_id=parent["id"],
            name="小明",
            grade_level="高中",
            weak_subjects=["数学", "物理"],
            learning_goals=["高考数学 120+", "物理保持 80+"],
            preferred_style="启发式",
            school="北京四中",
            notes="性格偏内向，需要老师耐心引导",
        )
        logger.info(f"创建默认学生: id={student_id}")
        return self.get_student(student_id)