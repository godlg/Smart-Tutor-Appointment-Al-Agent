"""会话历史服务(记忆可靠性 Phase 1)

职责:
- 统一读写:主控入口记录 human、完整回复结束记录 ai(可区分 partial),持久化到
  session_messages;同一会话顺序由 (session_id, seq) 唯一键保障。
- 可注入 provider:ConversationContext 通过本服务读同一 session 历史(而非各自
  维护私有 InMemoryChatMessageHistory),避免"同条消息两次"或"重复说的话被去重误删"。
- 恢复:重建服务 / 独立新进程后,按 seq 恢复历史,重复 restore 不产生重复消息。
- 摘要水位:滚动摘要经 conversation_repo 原子写入,重启不重复摘要。

本模块属于 services 层,仅依赖 db 层(config.identity 为叶子)。
"""

import re
import threading
from typing import Any, Dict, List, Optional

from config.identity import Identity, new_session_id
from db.repositories.identity_repository import SessionBindingError
from .identity_service import SessionOwnershipError, StudentClaimError
from .long_term_memory_service import LongTermMemoryService
from db.db_router import DatabaseRouter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


def strip_thought_output(text: str) -> str:
    """过滤路由/调试用的 [THOUGHT] 输出,保留用户实际看到的回复。

    按角色区段过滤,同一行紧接的 [REPLY]/[ERROR] 仍保留。
    无标签的普通回复保持原样,回复标记也保持前端可见形式。
    """
    if not text:
        return text
    cleaned = re.sub(
        r"\[THOUGHT\][^\r\n]*?(?=\[REPLY\]|\[ERROR\]|\r?\n|$)",
        "", text,
    )
    return "\n".join(line for line in cleaned.splitlines() if line.strip()).strip()


class ManagedChatHistory(InMemoryChatMessageHistory):
    """共享可见历史由入口唯一写入，局部字段提取 JSON 不进入此 provider。"""

    entry_managed: bool = True


class ConversationService:
    """会话历史 / 摘要水位 / 检查点持久化服务。"""

    def __init__(self, db_path: str = "sqlite:///data/smart_appointment.db",
                 db_router: Optional[DatabaseRouter] = None):
        self._owns_router = db_router is None
        self.db_router = db_router or DatabaseRouter(db_path)
        self.conversation_repo = self.db_router.conversation_repo
        self.memory_service = LongTermMemoryService(db_router=self.db_router)
        # 进程内 InMemory 历史缓存(session_id -> InMemoryChatMessageHistory),
        # 供 ConversationContext 直接读取;持久库是 source of truth,这里是热层。
        self._histories: Dict[str, InMemoryChatMessageHistory] = {}
        self._lock = threading.RLock()
        self._restored_sessions = set()
        self._history_watermarks: Dict[str, int] = {}

    def close(self) -> None:
        if self._owns_router:
            self.db_router.close()

    def ensure_session(self, identity: Identity) -> Identity:
        """Persist/check immutable user+student ownership, then return bound identity."""
        bound = identity if identity.session_id else identity.with_session(new_session_id())
        try:
            self.db_router.identity_repo.create_session(
                bound.session_id, bound.user_id, student_id=bound.student_id,
            )
        except SessionBindingError as exc:
            if exc.status_code == 409:
                raise StudentClaimError(str(exc), status_code=409) from exc
            raise SessionOwnershipError(str(exc)) from exc
        return bound

    # ------------------------------------------------------------------
    # 热层 InMemory 历史(供 ConversationContext 注入)
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """取(必要时创建)进程内历史对象;已存在则复用同一对象。"""
        with self._lock:
            hist = self._histories.get(session_id)
            if hist is None:
                hist = ManagedChatHistory()
                self._histories[session_id] = hist
            return hist

    def drop_history(self, session_id: str) -> None:
        """清理某 session 的进程内历史(缓存淘汰用)。持久库记录仍保留。"""
        with self._lock:
            self._histories.pop(session_id, None)
            self._restored_sessions.discard(session_id)
            self._history_watermarks.pop(session_id, None)

    # ------------------------------------------------------------------
    # 统一写入
    # ------------------------------------------------------------------

    def record_human(self, identity: Identity, content: str) -> int:
        """记录一条 human 消息(同时写持久库与热层),返回 seq。"""
        return self._record(identity, "human", content, status="complete")

    def record_ai(self, identity: Identity, content: str, partial: bool = False) -> int:
        """记录一条 ai 消息。partial=True 表示取消/异常,只保存已产生的可用内容。"""
        return self._record(
            identity, "ai", content, status="partial" if partial else "complete"
        )

    def _record(self, identity: Identity, role: str, content: str, status: str) -> int:
        identity = self.ensure_session(identity)
        seq = self.conversation_repo.append_message(
            session_id=identity.session_id,
            user_id=identity.user_id,
            student_id=identity.student_id,
            role=role,
            content=content,
            status=status,
        )
        # 重启/其它实例写入的缺口一并补齐；稳定热层只读取新 seq 后缀。
        self._sync_history(identity)
        return seq

    # ------------------------------------------------------------------
    # 恢复
    # ------------------------------------------------------------------

    def restore_history(self, identity: Identity) -> InMemoryChatMessageHistory:
        """从持久库恢复会话历史到热层(幂等:重复恢复不重复消息)。

        以持久库为准重建热层,因此重复调用不会追加重复消息。
        """
        identity = self.ensure_session(identity)
        return self._sync_history(identity, force=True)

    def _sync_history(self, identity: Identity, force: bool = False) -> InMemoryChatMessageHistory:
        """调用者已校验 ownership；显式恢复重建，正常消息写入同步新后缀。"""
        session_id = identity.session_id
        with self._lock:
            after_seq = 0 if force else self._history_watermarks.get(session_id, 0)
            rows = self.conversation_repo.get_messages(session_id, after_seq=after_seq)
            messages = [
                (HumanMessage if row["role"] == "human" else AIMessage)(
                    content=row["content"],
                    additional_kwargs={"seq": row["seq"], "status": row["status"]},
                )
                for row in rows
            ]
            hist = self.get_history(session_id)
            # 注入引用稳定是 provider 合约，此处同步拥有的缓存 list，不改消息对象。
            if force:
                hist.messages[:] = messages
            else:
                hist.messages.extend(messages)
            self._history_watermarks[session_id] = rows[-1]["seq"] if rows else after_seq
            self._restored_sessions.add(session_id)
            return hist

    def message_count(self, identity: Identity) -> int:
        identity = self.ensure_session(identity)
        return self.conversation_repo.count_messages(identity.session_id)

    def summary_watermark(self, identity: Identity) -> int:
        identity = self.ensure_session(identity)
        return self.conversation_repo.summary_watermark(identity.session_id)

    # ------------------------------------------------------------------
    # 滚动摘要(持久化水位)
    # ------------------------------------------------------------------

    def save_rollup(
        self, identity: Identity, start_seq: int, end_seq: int, content: str
    ) -> bool:
        """持久化一段滚动摘要(幂等)。返回 True 表示新写入。"""
        identity = self.ensure_session(identity)
        return self.memory_service.save_rollup_summary(
            session_id=identity.session_id,
            user_id=identity.user_id,
            student_id=identity.student_id,
            start_seq=start_seq,
            end_seq=end_seq,
            summary_text=content,
            auto_embed=False,
        )

    def get_summaries(self, identity: Identity) -> List[Dict[str, Any]]:
        identity = self.ensure_session(identity)
        return self.conversation_repo.get_summaries(identity.session_id)

    def recent_summary_text(self, identity: Identity, max_chars: int = 500) -> str:
        """取最近若干段摘要正文(供显式注入当前 prompt,不依赖向量 Top-k)。"""
        summaries = self.get_summaries(identity)
        if not summaries:
            return ""
        joined = "\n".join(s["content"] for s in summaries)
        return joined[-max_chars:]

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def save_checkpoint(self, identity: Identity, state: Optional[str],
                        payload: Optional[Dict[str, Any]] = None,
                        pending_ops: Optional[list] = None) -> None:
        import json as _json
        identity = self.ensure_session(identity)
        self.conversation_repo.save_checkpoint(
            session_id=identity.session_id,
            user_id=identity.user_id,
            student_id=identity.student_id,
            state=state,
            payload=_json.dumps(payload, ensure_ascii=False) if payload is not None else None,
            pending_ops=_json.dumps(pending_ops, ensure_ascii=False) if pending_ops is not None else None,
        )

    def load_checkpoint(self, identity: Identity) -> Optional[Dict[str, Any]]:
        identity = self.ensure_session(identity)
        return self.conversation_repo.load_checkpoint(identity.session_id)
