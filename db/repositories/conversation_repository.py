"""会话持久化数据访问对象(记忆可靠性 Phase 1)

职责:
- session_messages: 按 (session_id, seq) 顺序持久化对话消息(唯一键保障幂等)
- session_summaries: 滚动摘要水位记录((session_id,start_seq,end_seq) 唯一键幂等)
- session_checkpoints: 业务工作状态检查点(FlowState + payload + pending_ops)

只做数据访问,不做业务判定;返回 dict 便于 service 层使用。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from ..base.session_manager import SessionManager
from ..models import (
    SessionMessage, SessionSummary, SessionCheckpoint, ChatSession, LongTermMemory,
)
from .identity_repository import sqlite_write_scope, assert_binding, SessionBindingError


def _assert_known_owner(session, session_id: str, user_id: str,
                        student_id: Optional[str]) -> None:
    row = session.query(ChatSession).filter_by(session_id=session_id).first()
    if row is not None:
        assert_binding(row, user_id, student_id)


class ConversationRepository:
    """会话顺序消息 / 摘要水位 / 检查点数据访问对象。"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ------------------------------------------------------------------
    # 顺序消息
    # ------------------------------------------------------------------

    def next_seq(self, session_id: str) -> int:
        """会话内下一个顺序号(1-based)。"""
        with self.session_manager.session_scope() as session:
            max_seq = session.query(func.max(SessionMessage.seq)).filter(
                SessionMessage.session_id == session_id
            ).scalar()
            return int(max_seq or 0) + 1

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        status: str = "complete",
        seq: Optional[int] = None,
        student_id: Optional[str] = None,
    ) -> int:
        """Allocate and insert under SQLite's writer lock; only exact replays are idempotent."""
        if seq is not None and (not isinstance(seq, int) or isinstance(seq, bool) or seq < 1):
            raise ValueError("message seq 必须是正整数")
        with sqlite_write_scope(self.session_manager) as session:
            _assert_known_owner(session, session_id, user_id, student_id)
            if seq is None:
                max_seq = session.query(func.max(SessionMessage.seq)).filter(
                    SessionMessage.session_id == session_id
                ).scalar()
                seq = int(max_seq or 0) + 1
            existing = session.query(SessionMessage).filter(
                SessionMessage.session_id == session_id,
                SessionMessage.seq == seq,
            ).first()
            if existing is not None:
                recorded = (existing.user_id, existing.student_id, existing.role,
                            existing.content, existing.status)
                requested = (user_id, student_id, role, content, status)
                if recorded != requested:
                    raise ValueError("同一 message seq 已存在不同内容或 scope")
                return existing.seq
            session.add(SessionMessage(
                session_id=session_id,
                user_id=user_id,
                student_id=student_id,
                seq=seq,
                role=role,
                content=content,
                status=status,
            ))
            session.flush()
            return seq

    def get_messages(self, session_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        """按 seq 升序返回消息后缀；默认返回全部，兼容显式恢复。"""
        with self.session_manager.session_scope() as session:
            rows = session.query(SessionMessage).filter(
                SessionMessage.session_id == session_id,
                SessionMessage.seq > after_seq,
            ).order_by(SessionMessage.seq.asc()).all()
            return [
                {
                    "seq": r.seq,
                    "role": r.role,
                    "content": r.content,
                    "status": r.status,
                    "user_id": r.user_id,
                    "student_id": r.student_id,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def count_messages(self, session_id: str) -> int:
        with self.session_manager.session_scope() as session:
            return int(session.query(func.count(SessionMessage.id)).filter(
                SessionMessage.session_id == session_id
            ).scalar() or 0)

    # ------------------------------------------------------------------
    # 滚动摘要水位
    # ------------------------------------------------------------------

    def save_summary(
        self,
        session_id: str,
        user_id: str,
        start_seq: int,
        end_seq: int,
        content: str,
        student_id: Optional[str] = None,
    ) -> bool:
        """Legacy summary-only writer; new rollups use archive_rollup for both artifacts."""
        with sqlite_write_scope(self.session_manager) as session:
            _assert_known_owner(session, session_id, user_id, student_id)
            existing = session.query(SessionSummary).filter(
                SessionSummary.session_id == session_id,
                SessionSummary.start_seq == start_seq,
                SessionSummary.end_seq == end_seq,
            ).first()
            if existing is not None:
                return False
            session.add(SessionSummary(
                session_id=session_id,
                user_id=user_id,
                start_seq=start_seq,
                end_seq=end_seq,
                content=content,
                student_id=student_id,
            ))
            return True

    def archive_rollup(
        self, session_id: str, user_id: str, start_seq: int, end_seq: int,
        content: str, student_id: Optional[str] = None,
        importance: float = 0.5, embedding: Optional[List[float]] = None,
    ) -> bool:
        """Commit SessionSummary + LongTermMemory atomically, keyed by message span.

        No counter is stored separately: watermark is MAX(committed end_seq).
        A retry with different text/scope is a conflict, not silent success.
        """
        if (not isinstance(start_seq, int) or isinstance(start_seq, bool)
                or not isinstance(end_seq, int) or isinstance(end_seq, bool)
                or start_seq < 1 or end_seq < start_seq):
            raise ValueError("rollup span 必须是递增的正整数范围")
        with sqlite_write_scope(self.session_manager) as session:
            owner = session.query(ChatSession).filter_by(session_id=session_id).first()
            if owner is None:
                raise SessionBindingError("摘要归档要求可信会话归属")
            assert_binding(owner, user_id, student_id)
            span = dict(session_id=session_id, start_seq=start_seq, end_seq=end_seq)
            summary = session.query(SessionSummary).filter_by(**span).first()
            memory = session.query(LongTermMemory).filter_by(**span).first()
            if summary is not None or memory is not None:
                for artifact in (summary, memory):
                    if artifact is not None and (
                        artifact.user_id != user_id or artifact.student_id != student_id
                        or artifact.content != content
                    ):
                        raise ValueError("rollup span 已存在不同内容或 scope")
                if summary is not None and memory is not None:
                    return False
                raise ValueError("rollup span 存档不完整,拒绝推进水位")
            session.add(SessionSummary(
                **span, user_id=user_id, student_id=student_id, content=content,
            ))
            session.add(LongTermMemory(
                **span, user_id=user_id, student_id=student_id, content=content,
                memory_type="summary", importance=importance, embedding=embedding,
            ))
            session.flush()
        return True

    def get_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            rows = session.query(SessionSummary).filter(
                SessionSummary.session_id == session_id
            ).order_by(SessionSummary.start_seq.asc()).all()
            return [
                {
                    "start_seq": r.start_seq,
                    "end_seq": r.end_seq,
                    "content": r.content,
                    "user_id": r.user_id,
                    "student_id": r.student_id,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def summary_watermark(self, session_id: str) -> int:
        """已归档摘要的最大 end_seq(0 表示尚无摘要)。"""
        with self.session_manager.session_scope() as session:
            val = session.query(func.max(SessionSummary.end_seq)).filter(
                SessionSummary.session_id == session_id
            ).scalar()
            return int(val or 0)

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        session_id: str,
        user_id: str,
        state: Optional[str] = None,
        payload: Optional[str] = None,
        pending_ops: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> None:
        """Upsert checkpoint without ever changing its persisted owner."""
        with sqlite_write_scope(self.session_manager) as session:
            _assert_known_owner(session, session_id, user_id, student_id)
            row = session.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id == session_id
            ).first()
            if row is None:
                session.add(SessionCheckpoint(
                    session_id=session_id,
                    user_id=user_id,
                    state=state,
                    payload=payload,
                    pending_ops=pending_ops,
                ))
            else:
                if row.user_id != user_id:
                    raise SessionBindingError("检查点不属于当前身份")
                row.state = state
                if payload is not None:
                    row.payload = payload
                if pending_ops is not None:
                    row.pending_ops = pending_ops
                row.updated_at = datetime.utcnow()

    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            row = session.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id == session_id
            ).first()
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "user_id": row.user_id,
                "state": row.state,
                "payload": row.payload,
                "pending_ops": row.pending_ops,
                "updated_at": row.updated_at,
            }
