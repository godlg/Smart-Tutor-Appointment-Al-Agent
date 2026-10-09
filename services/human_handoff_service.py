"""人工接管服务（Human Handoff）—— 极端 case 兜底

背景:同一会话多次被分类器/下游 Agent "踢回"(unrelated) 后,家长可能陷入
"判无关 → 输出能力列表 → 再输入 → 又判无关" 的无限循环,没有任何转人工出口。
本服务提供三类兜底:

1. 主动关键词:家长输入含 `HANDOFF_KEYWORDS` 中任一子串 → 立即转人工
2. 踢回阈值:同会话累计踢回(record_kickback)达 `threshold`(默认 3,env 可配) → 自动转人工
3. 客服手动:客服 API 触发的记录(trigger_type='manual')

会话级计数放进程内 `_kickback_counts`(会话隔离,无需持久化);DB 只存触发记录,
客服可拉取 pending / 标记 resolved。

五层依赖:本模块属 services 层,只依赖 db 层。
"""

import json
import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional, Set

from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)

# 进程内"最近用户输入"缓冲的单会话容量(超出自动丢最旧)
RECENT_INPUTS_MAXLEN = 5

# 固定文案(勿改)
HANDOFF_REPLY = (
    "抱歉，连续几次没能帮您解决问题，已为您转接人工顾问老师。"
    "您的对话记录已保存，顾问老师会尽快与您联系。"
)
HANDED_OFF_REPLY = (
    "您的会话已转人工顾问处理中，请稍候。如需加急可拨打前台电话咨询。"
)


class HumanHandoffService:
    """人工接管服务:关键词识别 + 踢回计数 + 触发落库 + 客服处理。"""

    # 主动转人工关键词(子串匹配,命中任一即转)
    HANDOFF_KEYWORDS: List[str] = [
        "人工", "转人工", "真人", "人工客服", "人工服务",
        "客服", "找经理", "投诉", "人工顾问", "人工咨询", "人在吗",
    ]

    def __init__(
        self,
        db_path: str = "sqlite:///data/smart_appointment.db",
        threshold: Optional[int] = None,
    ):
        """构造参数全可选、向后兼容。

        Args:
            db_path: 数据库连接路径(与项目其他 service 一致)
            threshold: 踢回阈值;None 时读 env `HANDOFF_THRESHOLD`,再退回 3
        """
        self.db_router = DatabaseRouter(db_path)
        self.repo = self.db_router.handoff_repo
        self.threshold = self._resolve_threshold(threshold)
        # 进程内会话级踢回计数(无需持久化;DB 只存触发记录)
        self._kickback_counts: Dict[str, int] = {}
        # 已转人工的会话(内存快路径;DB pending 记录作为兜底)
        self._handed_off_sessions: Set[str] = set()
        # 进程内"最近用户输入"缓冲(会话隔离;无关轮也记录,转人工时补进快照)
        self._recent_inputs: Dict[str, deque] = {}

    @staticmethod
    def _resolve_threshold(threshold: Optional[int]) -> int:
        if threshold is not None:
            return int(threshold)
        raw = os.getenv("HANDOFF_THRESHOLD")
        try:
            return int(raw) if raw not in (None, "") else 3
        except (TypeError, ValueError):
            return 3

    # ------------------------------------------------------------------
    # 关键词
    # ------------------------------------------------------------------

    def match_keyword(self, task: str) -> Optional[str]:
        """返回 task 中命中的关键词;未命中返回 None。"""
        if not task:
            return None
        for kw in self.HANDOFF_KEYWORDS:
            if kw in task:
                return kw
        return None

    # ------------------------------------------------------------------
    # 踢回计数(会话隔离)
    # ------------------------------------------------------------------

    def record_kickback(self, session_id: str) -> bool:
        """踢回计数 +1,返回是否已达阈值(count >= threshold)。"""
        session_id = session_id or "default"
        count = self._kickback_counts.get(session_id, 0) + 1
        self._kickback_counts[session_id] = count
        return count >= self.threshold

    def get_count(self, session_id: str) -> int:
        """当前会话累计踢回数。"""
        return self._kickback_counts.get(session_id or "default", 0)

    # ------------------------------------------------------------------
    # 转人工状态
    # ------------------------------------------------------------------

    def is_handed_off(self, session_id: str) -> bool:
        """是否已转人工:内存 set 快路径 + DB pending 记录兜底。"""
        session_id = session_id or "default"
        if session_id in self._handed_off_sessions:
            return True
        try:
            if self.repo.get_pending_by_session(session_id):
                self._handed_off_sessions.add(session_id)
                return True
        except Exception as e:
            logger.warning(f"查询转人工状态失败,降级为内存判断: {e}")
        return False

    def trigger_handoff(
        self,
        session_id: str,
        trigger_type: str,
        reason: Optional[str] = None,
        unrelated_count: int = 0,
        transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        """触发转人工(幂等):同 session 已有 pending 记录则不重复建,返回已有记录。

        transcript: 触发时的会话历史快照(JSON 字符串),可空。
        """
        session_id = session_id or "default"
        existing = self.repo.get_pending_by_session(session_id)
        if existing:
            self._handed_off_sessions.add(session_id)
            return existing
        handoff_id = self.repo.create(
            session_id=session_id,
            trigger_type=trigger_type,
            trigger_reason=reason,
            unrelated_count=unrelated_count,
            transcript=transcript,
        )
        self._handed_off_sessions.add(session_id)
        return self.repo.get_by_id(handoff_id)

    # ------------------------------------------------------------------
    # 客服处理
    # ------------------------------------------------------------------

    def get_pending(self) -> List[Dict[str, Any]]:
        """所有待处理记录。"""
        return self.repo.get_pending()

    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """全部记录(可选按 status 过滤)。"""
        return self.repo.get_all(status)

    def resolve(
        self,
        handoff_id: int,
        handled_by: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        """标记记录已处理;不存在返回 False。"""
        ok = self.repo.resolve(handoff_id, handled_by=handled_by, note=note)
        if ok:
            record = self.repo.get_by_id(handoff_id)
            if record and record.get("session_id") in self._handed_off_sessions:
                # 已处理 → 解除内存短路,允许该会话继续正常对话
                self._handed_off_sessions.discard(record["session_id"])
        return ok

    def claim(self, handoff_id: int, agent_id: str) -> Optional[Dict[str, Any]]:
        """客服认领工单;记录不存在返回 None。已认领时幂等(保留原客服)。"""
        return self.repo.claim(handoff_id, agent_id)

    def agent_reply(
        self,
        handoff_id: int,
        agent_id: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """客服回复:仅记录存在且仍 pending 时落库(role='agent')。

        记录不存在 / 已 resolve → 返回 None(不落库)。
        """
        record = self.repo.get_by_id(handoff_id)
        if not record or record.get("status") != "pending":
            return None
        message_id = self.repo.add_message(
            handoff_id=handoff_id,
            session_id=record["session_id"],
            role="agent",
            content=content,
        )
        messages = self.repo.get_messages(handoff_id)
        for message in messages:
            if message["id"] == message_id:
                return message
        return None

    def parent_message(
        self,
        session_id: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """家长消息:该 session 有 pending 转人工记录时落库(role='parent')。

        无 pending 记录 → 返回 None(不抛异常)。
        """
        record = self.repo.get_pending_by_session(session_id)
        if not record:
            return None
        message_id = self.repo.add_message(
            handoff_id=record["id"],
            session_id=session_id,
            role="parent",
            content=content,
        )
        messages = self.repo.get_messages(record["id"])
        for message in messages:
            if message["id"] == message_id:
                return message
        return None

    def get_messages(self, handoff_id: int) -> List[Dict[str, Any]]:
        """某工单全部消息(升序)。"""
        return self.repo.get_messages(handoff_id)

    def get_messages_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """某 session 全部消息(升序,供家长侧轮询)。"""
        return self.repo.get_messages_by_session(session_id)

    def get_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """该 session 最新一条记录(不限 status)。"""
        return self.repo.get_by_session(session_id)

    # ------------------------------------------------------------------
    # 最近用户输入缓冲 + transcript 组装
    # ------------------------------------------------------------------

    def _ensure_recent_inputs(self) -> Dict[str, deque]:
        """返回最近输入缓冲;对 __new__ 手搓实例惰性补建(测试/demo 兼容)。"""
        buf = getattr(self, "_recent_inputs", None)
        if buf is None:
            buf = {}
            self._recent_inputs = buf
        return buf

    def record_input(self, session_id: str, content: str) -> None:
        """记录一条用户输入(会话隔离;超出容量自动丢最旧)。

        所有 HTTP 用户输入进主控时都会调用,包含被判 unrelated、只 yield
        能力列表而不写会话历史的无关轮 —— 转人工时这些原话仍可进快照。
        """
        if content is None:
            return
        self._ensure_recent_inputs().setdefault(
            session_id or "default", deque(maxlen=RECENT_INPUTS_MAXLEN)
        ).append(content)

    def build_transcript(self, session_id: str, history_messages) -> Optional[str]:
        """会话历史快照 + 未进历史的最近用户输入(无关轮兜底)。

        snapshot_transcript 只吃 history.messages;无关轮(判 unrelated →
        仅 yield 能力列表)不写会话历史,导致工单 transcript 缺失家长原话。
        这里把缓冲中"未在历史里出现"的用户输入补进快照,按 content 去重
        (业务轮既在 history 又进缓冲时不重复)。history 为 None 时按空处理。
        结果非空返回 JSON 字符串,否则 None。
        """
        snapshot: List[Dict[str, Any]] = []
        base = self.snapshot_transcript(history_messages)
        if base:
            snapshot = json.loads(base)
        seen = {item.get("content") for item in snapshot}
        for content in self._ensure_recent_inputs().get(session_id or "default", []):
            if content in seen:
                continue
            snapshot.append({"role": "human", "content": content})
            seen.add(content)
        if not snapshot:
            return None
        return json.dumps(snapshot, ensure_ascii=False)

    @staticmethod
    def snapshot_transcript(messages) -> Optional[str]:
        """把 langchain message 列表转 JSON 字符串快照。

        输出 `[{"role": <type>, "content": <content>}, ...]`;
        空列表 / None 返回 None。无法识别 type 时 role 用 "unknown",content 用 str()。
        """
        if not messages:
            return None
        snapshot: List[Dict[str, Any]] = []
        for m in messages:
            role = getattr(m, "type", None) or "unknown"
            raw = getattr(m, "content", None)
            content = raw if isinstance(raw, str) else str(raw)
            snapshot.append({"role": role, "content": content})
        return json.dumps(snapshot, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 固定文案
    # ------------------------------------------------------------------

    @staticmethod
    def handoff_reply() -> str:
        """触发轮文案(keyword / kickback_threshold 共用)。"""
        return HANDOFF_REPLY

    @staticmethod
    def handed_off_reply() -> str:
        """已转人工短路轮文案。"""
        return HANDED_OFF_REPLY
