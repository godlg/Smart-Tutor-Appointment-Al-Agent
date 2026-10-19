from .base import SessionManager
from .base.session_manager import DEFAULT_DB_PATH  # 库路径单一来源,避免字面量漂移
from .repositories import (
    KnowledgeRepository,
    UserBehaviorRepository,
    TeacherRepository,
    StudentRepository,
    LessonRepository,
    PackageRepository,
    ProgressRepository,
    LongTermMemoryRepository,
    HumanHandoffRepository,
    IdentityRepository,
    ConversationRepository,
)
from typing import Optional


class DatabaseRouter:
    """
    数据库路由器（家教场景）

    统一管理数据库连接和会话，提供各领域 Repository 访问入口。
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.session_manager = SessionManager(db_path)

        # 领域 Repository (P1-9: 统一 *_repo 后缀命名)
        self.knowledge_repo = KnowledgeRepository(self.session_manager)
        self.user_behavior_repo = UserBehaviorRepository(self.session_manager)
        self.teacher_repo = TeacherRepository(self.session_manager)
        self.student_repo = StudentRepository(self.session_manager)
        self.lesson_repo = LessonRepository(self.session_manager)
        self.package_repo = PackageRepository(self.session_manager)
        self.progress_repo = ProgressRepository(self.session_manager)
        self.memory_repo = LongTermMemoryRepository(self.session_manager)
        self.handoff_repo = HumanHandoffRepository(self.session_manager)
        # 记忆可靠性 Phase 1:匿名身份 / 会话归属 / 顺序消息 / 摘要 / 检查点
        self.identity_repo = IdentityRepository(self.session_manager)
        self.conversation_repo = ConversationRepository(self.session_manager)

    # 保留少量短 alias (services 层大量使用, 重命名成本太高)
    # 仅保留复数语义清晰的短名. knowledge alias 已删除 (P1-9)
    @property
    def user_behavior(self) -> UserBehaviorRepository:
        return self.user_behavior_repo

    @property
    def teachers(self) -> TeacherRepository:
        return self.teacher_repo

    @property
    def students(self) -> StudentRepository:
        return self.student_repo

    @property
    def lessons(self) -> LessonRepository:
        return self.lesson_repo

    @property
    def packages(self) -> PackageRepository:
        return self.package_repo

    @property
    def progress(self) -> ProgressRepository:
        return self.progress_repo

    def close(self):
        self.session_manager.close()