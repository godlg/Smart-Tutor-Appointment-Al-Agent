"""身份 / 会话归属 数据访问对象(记忆可靠性 Phase 1)

职责:
- identities: 服务端签发的匿名身份
- identity_students: 身份 → 学员 namespace 的认领关系(全局唯一)
- chat_sessions: 会话 → 归属身份(ownership)

返回 dict 或原始值,供 service 层做业务判定(不在此抛业务异常)。
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from ..base.session_manager import SessionManager
from ..models import (
    IdentityRecord, IdentityStudent, ChatSession, SessionMessage, SessionSummary,
    SessionCheckpoint, LongTermMemory,
)


class SessionBindingError(ValueError):
    """Repository-level fail-closed ownership conflict, translated by services."""
    status_code = 403

    def __init__(self, message: str, status_code: int = 403):
        super().__init__(message)
        self.status_code = status_code


@contextmanager
def sqlite_write_scope(session_manager: SessionManager):
    """Lock before reading MAX/ownership; allocation and write share one transaction."""
    with session_manager.session_scope() as session:
        if session_manager.engine.dialect.name == "sqlite":
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        yield session


def assert_binding(row: ChatSession, user_id: str, student_id: Optional[str]) -> None:
    if not row.user_id or row.user_id != user_id:
        raise SessionBindingError("会话不属于当前身份")
    if row.student_id != student_id:
        raise SessionBindingError("会话学员 scope 不可更改", status_code=409)


def assert_artifacts_scoped(session, session_id: str, user_id: str,
                            student_id: Optional[str]) -> None:
    """Do not expose inconsistent/unknown legacy rows even when an owner row exists."""
    for model in (SessionMessage, SessionSummary, SessionCheckpoint, LongTermMemory):
        scoped = session.query(model.id).filter(model.session_id == session_id)
        if scoped.filter(or_(model.user_id.is_(None), model.user_id != user_id)).first():
            raise SessionBindingError("会话存档归属不一致,拒绝访问")
        if hasattr(model, "student_id"):
            mismatch = (model.student_id.is_not(None) if student_id is None else
                        or_(model.student_id.is_(None), model.student_id != student_id))
            if scoped.filter(mismatch).first():
                raise SessionBindingError("会话存档学员 scope 不一致", status_code=409)


class IdentityRepository:
    """身份 / 会话归属数据访问对象。"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ------------------------------------------------------------------
    # 匿名身份
    # ------------------------------------------------------------------

    def create_identity(self, user_id: str) -> str:
        """登记一个新的匿名身份(已存在则幂等返回)。"""
        with self.session_manager.session_scope() as session:
            existing = session.query(IdentityRecord).filter(
                IdentityRecord.user_id == user_id
            ).first()
            if existing is None:
                session.add(IdentityRecord(user_id=user_id))
            return user_id

    def identity_exists(self, user_id: str) -> bool:
        """该 user_id 是否为本服务签发过(用于校验 cookie,不盲信客户端)。"""
        if not user_id:
            return False
        with self.session_manager.session_scope() as session:
            return session.query(IdentityRecord).filter(
                IdentityRecord.user_id == user_id
            ).first() is not None

    # ------------------------------------------------------------------
    # 会话归属
    # ------------------------------------------------------------------

    def get_session_owner(self, session_id: str) -> Optional[str]:
        """返回会话归属的 user_id;不存在返回 None。"""
        if not session_id:
            return None
        with self.session_manager.session_scope() as session:
            row = session.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            return row.user_id if row else None

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            row = session.query(ChatSession).filter_by(session_id=session_id).first()
            if row is None:
                return None
            return {"session_id": row.session_id, "user_id": row.user_id,
                    "student_id": row.student_id}

    def create_session(self, session_id: str, user_id: str,
                       student_id: Optional[str] = None) -> str:
        """Atomic claim/check; no caller preflight can authorize a conflicting owner."""
        if not session_id or not user_id:
            raise SessionBindingError("缺少会话或身份")
        try:
            with sqlite_write_scope(self.session_manager) as session:
                row = session.query(ChatSession).filter_by(session_id=session_id).first()
                if row is not None:
                    assert_binding(row, user_id, student_id)
                    assert_artifacts_scoped(session, session_id, user_id, student_id)
                else:
                    for model in (SessionMessage, SessionSummary, SessionCheckpoint, LongTermMemory):
                        if session.query(model.id).filter_by(session_id=session_id).first():
                            raise SessionBindingError("已有存档但没有可信会话归属,拒绝认领")
                if student_id is not None:
                    self._claim_student_in_session(session, user_id, student_id)
                if row is None:
                    session.add(ChatSession(session_id=session_id, user_id=user_id,
                                            student_id=student_id))
                session.flush()
            return session_id
        except IntegrityError as exc:
            # Unique collision must be rechecked after rollback, never called success.
            row = self.get_session(session_id)
            if row is None:
                raise SessionBindingError("未知会话绑定冲突,拒绝访问") from exc
            if row["user_id"] != user_id:
                raise SessionBindingError("会话不属于当前身份") from exc
            if row["student_id"] != student_id:
                raise SessionBindingError("会话学员 scope 不可更改", 409) from exc
            # Other integrity failures are ambiguous: fail closed even for matching owner.
            raise SessionBindingError("无法确认会话绑定冲突,拒绝访问") from exc

    @staticmethod
    def _claim_student_in_session(session, user_id: str, student_id: str) -> None:
        row = session.query(IdentityStudent).filter_by(student_id=student_id).first()
        if row is not None and row.user_id != user_id:
            raise SessionBindingError("学员 namespace 已被其它身份认领", 409)
        if row is None:
            session.add(IdentityStudent(user_id=user_id, student_id=student_id))

    def list_sessions_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        with self.session_manager.session_scope() as session:
            rows = session.query(ChatSession).filter(
                ChatSession.user_id == user_id
            ).all()
            return [
                {"session_id": r.session_id, "user_id": r.user_id,
                 "created_at": r.created_at}
                for r in rows
            ]

    # ------------------------------------------------------------------
    # 学员 namespace 认领
    # ------------------------------------------------------------------

    def get_student_owner(self, student_id: str) -> Optional[str]:
        """返回学员 namespace 被哪个身份认领;未认领返回 None。"""
        if not student_id:
            return None
        with self.session_manager.session_scope() as session:
            row = session.query(IdentityStudent).filter(
                IdentityStudent.student_id == student_id
            ).first()
            return row.user_id if row else None

    def claim_student(self, user_id: str, student_id: str) -> bool:
        """允许认领(不存在该 student 或已被同一 user_id 认领)。冲突返回 False。"""
        try:
            with sqlite_write_scope(self.session_manager) as session:
                self._claim_student_in_session(session, user_id, student_id)
                session.flush()
            return True
        except SessionBindingError:
            return False
        except IntegrityError:
            return self.get_student_owner(student_id) == user_id
