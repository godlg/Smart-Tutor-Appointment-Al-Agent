"""
进度记录服务层
"""

import logging
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class ProgressService:
    """课后进度业务服务"""

    def __init__(self):
        self.db = DatabaseRouter()

    @property
    def progress_repo(self):
        return self.db.progress

    def add_progress(self, **kwargs) -> int:
        return self.progress_repo.add_progress(**kwargs)

    def get_progress(self, progress_id: int) -> Optional[Dict[str, Any]]:
        return self.progress_repo.get_progress_by_id(progress_id)

    def get_progress_by_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        return self.progress_repo.get_progress_by_lesson(lesson_id)

    def get_student_progress(self, student_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        return self.progress_repo.get_progress_by_student(student_id, limit)

    def update_progress(self, progress_id: int, **kwargs) -> bool:
        return self.progress_repo.update_progress(progress_id, **kwargs)