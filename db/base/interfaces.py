"""
数据访问抽象接口（家教场景）

定义 Teacher / Lesson / KnowledgeBase / UserBehavior 等核心实体的数据访问接口。
Repository 实现位于 db/repositories/，上层 Service 通过接口与实现解耦。
"""


"""
只定义“做什么”,不定义“怎么做”
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime




#教师管理
class BaseTeacherRepository(ABC):
    """老师数据访问抽象接口（家教场景）"""

#添加教师
    @abstractmethod
    def add_teacher(self, **kwargs) -> int:
        pass

    @abstractmethod
    def get_teacher_by_id(self, teacher_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_teacher_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_teachers(self, only_active: bool = True) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def find_teachers_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def find_teachers_by_grade(self, grade_level: str) -> List[Dict[str, Any]]:
        pass


#场景，用户问“有没有擅长高一数学的老师”
    @abstractmethod
    def find_teachers_by_subject_and_grade(
        self, subject: str, grade_level: str
    ) -> List[Dict[str, Any]]:
        pass



#课时仓储
class BaseLessonRepository(ABC):
    """课时数据访问抽象接口（家教场景）

    同时覆盖：单次课 / 周固定课 / 调课 / 退课 / 试讲反馈。
    """

#添加课程
    @abstractmethod
    def add_lesson(self, **kwargs) -> int:
        pass

    @abstractmethod
    def get_lesson(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_lessons_by_student(self, student_id: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_lessons_by_teacher(
        self, teacher_id: int, start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        pass

#排课冲突检测
    @abstractmethod
    def check_teacher_conflict(
        self, teacher_id: int, start_time: datetime, end_time: datetime,
    ) -> bool:
        pass

    @abstractmethod
    def update_lesson(self, lesson_id: int, **updates) -> bool:
        pass

    @abstractmethod
    def delete_lesson(self, lesson_id: int) -> bool:
        pass



#知识库仓储
class BaseKnowledgeRepository(ABC):
    """知识库数据访问抽象接口（家教场景）

    知识分类：course_intro / subject_method / teacher_intro /
            exam_strategy / parent_guide / faq
    """


#支持向量字段
    @abstractmethod
    def add_document(
        self, content: str, category: str,
        keywords: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
        chunk_index: Optional[int] = None,
        source_hash: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> int:
        pass

    @abstractmethod
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_documents(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        pass

#任意文档可更新
    @abstractmethod
    def update_document(
        self, doc_id: int, content: Optional[str] = None,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> bool:
        pass

#默认软删除
    @abstractmethod
    def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        pass

    @abstractmethod
    def search_documents_by_category(self, category: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def search_documents_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def search_documents_with_filters(
        self,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """元数据过滤查询:用 SQL WHERE 硬匹配缩候选集。

        设计目的:为向量检索前提供业务硬筛选,
        避免"用户问初二数学,被高三数学卡位"。

        Args:
            subject: 学科(如 "数学");None 表示不筛选
            grade_level: 年级段(如 "初中");None 表示不筛选
            doc_type: 文档类型(如 "course_intro");None 表示不筛选

        Returns:
            满足全部非 None 过滤条件的活跃文档列表
        """
        pass

    @abstractmethod
    def get_all_categories(self) -> List[str]:
        pass

    @abstractmethod
    def get_documents_count(self) -> int:
        pass

    @abstractmethod
    def find_chunks_by_source_hash(
        self, source_hash: str,
    ) -> List[Dict[str, Any]]:
        """按文件 SHA256 查所有 chunk,按 chunk_index ASC 排序。

        P1-13:用于上传去重 —— 同 source_hash 已存在则视为重复,
        直接返回已有 doc_ids (skipped=True),避免重复入库。
        不含 is_active=0 的软删除行。
        """
        pass


class BaseUserBehaviorRepository(ABC):
    """用户行为数据访问抽象接口（家教场景）"""

    @abstractmethod
    def record_behavior(
        self, user_id: str, action_type: str,
        action_data: Optional[Dict[str, Any]] = None,
        teacher_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> int:
        pass

    @abstractmethod
    def get_user_behaviors(
        self, user_id: str, action_type: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_user_preference(
        self, user_id: str, preference_type: str, preference_value: str,
    ) -> bool:
        pass

    @abstractmethod
    def get_user_preferences(
        self, user_id: str, preference_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def create_recommendation(
        self, user_id: str, recommendation_type: str, content: str,
        teacher_id: Optional[int] = None,
    ) -> int:
        pass

    @abstractmethod
    def get_pending_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def mark_recommendation_sent(self, recommendation_id: int) -> bool:
        pass

    @abstractmethod
    def get_user_statistics(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        pass


class BaseLongTermMemoryRepository(ABC):
    """长期记忆数据访问抽象接口（分层记忆增强）

    内容型长期记忆：咨询摘要 / 预约事件 / 偏好,每条带 embedding + importance + 时间戳。
    """

    @abstractmethod
    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        session_id: Optional[str] = None,
    ) -> int:
        """新增一条长期记忆,返回新记录 id。"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """按 id 查询单条记忆;不存在返回 None。"""
        pass

    @abstractmethod
    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按用户(可选按类型)拉取记忆,按 created_at 倒序。"""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新记忆内容 / 向量 / 重要度;不存在返回 False。"""
        pass

    @abstractmethod
    def touch_memory(self, memory_id: int) -> bool:
        """刷新 last_accessed_at。"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: int) -> bool:
        """删除单条记忆;不存在返回 False。"""
        pass
