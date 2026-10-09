"""长期记忆数据访问对象（分层记忆增强）

职责：long_term_memories 表的增 / 查 / 更新 / 删除 / 召回候选集读取。
所有方法返回 dict(embedding 已从 JSON 反序列化为 list),便于 service 层直接算相似度。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base.interfaces import BaseLongTermMemoryRepository
from ..base.session_manager import SessionManager
from ..models import LongTermMemory


class LongTermMemoryRepository(BaseLongTermMemoryRepository):
    """长期记忆数据访问对象。"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> int:
        """新增一条长期记忆,返回新记录 id。"""
        with self.session_manager.session_scope() as session:
            memory = LongTermMemory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                embedding=embedding,
                importance=importance,
                session_id=session_id,
                student_id=student_id,
            )
            session.add(memory)
            session.flush()
            return memory.id

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """按 id 查询单条记忆;不存在返回 None。"""
        with self.session_manager.session_scope() as session:
            memory = session.query(LongTermMemory).filter(
                LongTermMemory.id == memory_id
            ).first()
            if memory is None:
                return None
            return self._to_dict(memory)

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
        student_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按用户(可选按类型)拉取记忆,按 created_at 倒序。

        limit=None 时返回全部,供召回评分在内存里算 Top-5。
        """
        with self.session_manager.session_scope() as session:
            query = session.query(LongTermMemory).filter(
                LongTermMemory.user_id == user_id,
                LongTermMemory.student_id == student_id,
            )
            if memory_type is not None:
                query = query.filter(LongTermMemory.memory_type == memory_type)
            query = query.order_by(LongTermMemory.created_at.desc())
            if limit is not None:
                query = query.limit(limit)
            memories = query.all()
            return [self._to_dict(m) for m in memories]

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新记忆内容 / 向量 / 重要度;不存在返回 False。"""
        with self.session_manager.session_scope() as session:
            memory = session.query(LongTermMemory).filter(
                LongTermMemory.id == memory_id
            ).first()
            if memory is None:
                return False
            if content is not None:
                memory.content = content
            if embedding is not None:
                memory.embedding = embedding
            if importance is not None:
                memory.importance = importance
            return True

    def touch_memory(self, memory_id: int) -> bool:
        """刷新 last_accessed_at(召回命中后调用,支持时间衰减的活跃度反馈)。"""
        with self.session_manager.session_scope() as session:
            memory = session.query(LongTermMemory).filter(
                LongTermMemory.id == memory_id
            ).first()
            if memory is None:
                return False
            memory.last_accessed_at = datetime.utcnow()
            return True

    def delete_memory(self, memory_id: int) -> bool:
        """删除单条记忆;不存在返回 False。"""
        with self.session_manager.session_scope() as session:
            memory = session.query(LongTermMemory).filter(
                LongTermMemory.id == memory_id
            ).first()
            if memory is None:
                return False
            session.delete(memory)
            return True

    def _to_dict(self, memory: LongTermMemory) -> Dict[str, Any]:
        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "student_id": memory.student_id,
            "start_seq": memory.start_seq,
            "end_seq": memory.end_seq,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "embedding": memory.embedding,
            "importance": memory.importance,
            "session_id": memory.session_id,
            "created_at": memory.created_at,
            "last_accessed_at": memory.last_accessed_at,
        }
