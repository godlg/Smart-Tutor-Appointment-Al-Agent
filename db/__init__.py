"""
Database Module（家教场景）
"""

from .db_router import DatabaseRouter
from .repositories import (
    KnowledgeRepository,
    UserBehaviorRepository,
    TeacherRepository,
    StudentRepository,
    LessonRepository,
    PackageRepository,
    ProgressRepository,
    LongTermMemoryRepository,
)
from .base import SessionManager
from .models import (
    Base,
    SubjectEnum,
    GradeLevelEnum,
    TeachingStyleEnum,
    LessonTypeEnum,
    LessonStatusEnum,
    KnowledgeCategoryEnum,
    # 家教核心模型
    Teacher,
    Parent,
    StudentProfile,
    LearningGoal,
    CoursePackage,
    LessonSeries,
    Lesson,
    ProgressRecord,
    TrialFeedback,
    # 通用
    KnowledgeDocument,
    UserBehavior,
    UserPreference,
    UserRecommendation,
    # 长期记忆（分层记忆增强）
    LongTermMemory,
    # 人工接管
    HumanHandoff,
    HumanHandoffMessage,
    # 记忆可靠性 Phase 1
    IdentityRecord,
    IdentityStudent,
    ChatSession,
    SessionMessage,
    SessionSummary,
    SessionCheckpoint,
)

__all__ = [
    # 主要入口
    "DatabaseRouter",
    # Repository
    "KnowledgeRepository",
    "UserBehaviorRepository",
    "TeacherRepository",
    "StudentRepository",
    "LessonRepository",
    "PackageRepository",
    "ProgressRepository",
    "LongTermMemoryRepository",
    # 基础设施
    "SessionManager",
    "Base",
    # 家教模型
    "Teacher",
    "Parent",
    "StudentProfile",
    "LearningGoal",
    "CoursePackage",
    "LessonSeries",
    "Lesson",
    "ProgressRecord",
    "TrialFeedback",
    # 通用模型
    "KnowledgeDocument",
    "UserBehavior",
    "UserPreference",
    "UserRecommendation",
    "LongTermMemory",
    # 人工接管
    "HumanHandoff",
    "HumanHandoffMessage",
    # 枚举
    "SubjectEnum",
    "GradeLevelEnum",
    "TeachingStyleEnum",
    "LessonTypeEnum",
    "LessonStatusEnum",
    "KnowledgeCategoryEnum",
]