"""
Database Base Module（家教场景）

数据库基础模块，包含：
- 会话管理（SessionManager）
- 抽象接口（BaseTeacherRepository / BaseLessonRepository /
            BaseKnowledgeRepository / BaseUserBehaviorRepository）
"""

from .session_manager import SessionManager
from .interfaces import (
    BaseTeacherRepository,
    BaseLessonRepository,
    BaseKnowledgeRepository,
    BaseUserBehaviorRepository,
    BaseLongTermMemoryRepository,
)


#公共api声明
#_all_是 python的白名单机制
#sessionmanager管理数据库会话，BaseTeacherRepository是教师数据访问的抽象接口，
#BaseLessonRepository是课程数据访问的抽象接口
#BaseKnowledgeReposity是知识库数据访问的抽象接口
#baseUserBehaviorReposity是用户行为数据访问的抽象接口
__all__ = [
    'SessionManager',
    'BaseTeacherRepository',
    'BaseLessonRepository',
    'BaseKnowledgeRepository',
    'BaseUserBehaviorRepository',
    'BaseLongTermMemoryRepository',
]
