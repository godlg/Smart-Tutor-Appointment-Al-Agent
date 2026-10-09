"""人工接管数据访问对象

职责:human_handoffs 表的增 / 查 / 标记已处理。
所有方法返回 dict,便于 service 层直接使用(与本仓库其他 Repository 一致)。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base.session_manager import SessionManager
from ..models import HumanHandoff, HumanHandoffMessage


class HumanHandoffRepository:
    """人工接管数据访问对象。"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def create(
        self,
        session_id: str,
        trigger_type: str,
        trigger_reason: Optional[str] = None,
        unrelated_count: int = 0,
        transcript: Optional[str] = None,
    ) -> int:
        """新增一条转人工记录,返回新记录 id。

        transcript: 触发时的会话历史快照(JSON 字符串),可空。
        """
        with self.session_manager.session_scope() as session:
            record = HumanHandoff(
                session_id=session_id,
                trigger_type=trigger_type,
                trigger_reason=trigger_reason,
                unrelated_count=unrelated_count,
                status="pending",
                transcript=transcript,
            )
            session.add(record)
            session.flush()
            return record.id

    def get_by_id(self, handoff_id: int) -> Optional[Dict[str, Any]]:
        """按 id 查询单条记录;不存在返回 None。"""
        with self.session_manager.session_scope() as session:
            record = session.query(HumanHandoff).filter(
                HumanHandoff.id == handoff_id
            ).first()
            if record is None:
                return None
            return self._to_dict(record)

    def get_pending_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """按会话查最新一条 pending 记录;不存在返回 None。"""
        with self.session_manager.session_scope() as session:
            record = session.query(HumanHandoff).filter(
                HumanHandoff.session_id == session_id,
                HumanHandoff.status == "pending",
            ).order_by(HumanHandoff.created_at.desc()).first()
            if record is None:
                return None
            return self._to_dict(record)

    def get_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """按会话查最新一条记录(不限 status);不存在返回 None。"""
        with self.session_manager.session_scope() as session:
            record = session.query(HumanHandoff).filter(
                HumanHandoff.session_id == session_id,
            ).order_by(HumanHandoff.created_at.desc()).first()
            if record is None:
                return None
            return self._to_dict(record)

    def get_pending(self) -> List[Dict[str, Any]]:
        """所有 pending 记录,按 created_at 倒序。"""
        return self.get_all(status="pending")

    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """全部记录(可选按 status 过滤),按 created_at 倒序。"""
        with self.session_manager.session_scope() as session:
            query = session.query(HumanHandoff)
            if status is not None:
                query = query.filter(HumanHandoff.status == status)
            records = query.order_by(HumanHandoff.created_at.desc()).all()
            return [self._to_dict(r) for r in records]

    def resolve(
        self,
        handoff_id: int,
        handled_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        """把记录标记为已处理(resolved + resolved_at);不存在返回 False。"""
        with self.session_manager.session_scope() as session:
            record = session.query(HumanHandoff).filter(
                HumanHandoff.id == handoff_id
            ).first()
            if record is None:
                return False
            record.status = "resolved"
            record.handled_by = handled_by
            record.resolve_note = note
            record.resolved_at = datetime.utcnow()
            return True

    def claim(self, handoff_id: int, agent_id: str) -> Optional[Dict[str, Any]]:
        """认领工单:置 agent_id。

        已认领(agent_id 非空)时幂等,返回现有记录(不覆盖原客服);
        记录不存在返回 None。
        """
        with self.session_manager.session_scope() as session:
            record = session.query(HumanHandoff).filter(
                HumanHandoff.id == handoff_id
            ).first()
            if record is None:
                return None
            if not record.agent_id:
                record.agent_id = agent_id
            return self._to_dict(record)

    def add_message(
        self,
        handoff_id: int,
        session_id: str,
        role: str,
        content: str,
    ) -> int:
        """追加一条工单消息,返回新消息 id。"""
        with self.session_manager.session_scope() as session:
            message = HumanHandoffMessage(
                handoff_id=handoff_id,
                session_id=session_id,
                role=role,
                content=content,
            )
            session.add(message)
            session.flush()
            return message.id

    def get_messages(self, handoff_id: int) -> List[Dict[str, Any]]:
        """某工单的全部消息,按 created_at 升序(同刻按 id 升序)。"""
        with self.session_manager.session_scope() as session:
            records = session.query(HumanHandoffMessage).filter(
                HumanHandoffMessage.handoff_id == handoff_id,
            ).order_by(
                HumanHandoffMessage.created_at.asc(),
                HumanHandoffMessage.id.asc(),
            ).all()
            return [self._message_to_dict(r) for r in records]

    def get_messages_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """某 session 的全部消息,按 created_at 升序(家长侧轮询)。"""
        with self.session_manager.session_scope() as session:
            records = session.query(HumanHandoffMessage).filter(
                HumanHandoffMessage.session_id == session_id,
            ).order_by(
                HumanHandoffMessage.created_at.asc(),
                HumanHandoffMessage.id.asc(),
            ).all()
            return [self._message_to_dict(r) for r in records]

    @staticmethod
    def _to_dict(record: HumanHandoff) -> Dict[str, Any]:
        return {
            "id": record.id,
            "session_id": record.session_id,
            "trigger_type": record.trigger_type,
            "trigger_reason": record.trigger_reason,
            "unrelated_count": record.unrelated_count,
            "status": record.status,
            "handled_by": record.handled_by,
            "resolve_note": record.resolve_note,
            "transcript": record.transcript,
            "agent_id": record.agent_id,
            "created_at": record.created_at,
            "resolved_at": record.resolved_at,
        }

    @staticmethod
    def _message_to_dict(record: HumanHandoffMessage) -> Dict[str, Any]:
        return {
            "id": record.id,
            "handoff_id": record.handoff_id,
            "session_id": record.session_id,
            "role": record.role,
            "content": record.content,
            "created_at": record.created_at,
        }
