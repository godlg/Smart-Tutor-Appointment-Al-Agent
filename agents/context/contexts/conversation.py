"""
会话历史上下文

包装 langchain_core.chat_history.InMemoryChatMessageHistory,
按 session_id 缓存用户 / 顾问历史消息,并把历史渲染成纯文本注入,
让 LLM 把握多轮对话的来龙去脉(与 Trial / 正式课 的 chats_by_session_id 同模式)。
"""

from typing import Dict

from langchain_core.chat_history import InMemoryChatMessageHistory

from ..base import BaseContext
from ..types import PRIORITY_CONVERSATION

# 无历史时的占位文案
_NO_HISTORY = "暂无历史消息"
# 折叠更早对话时用的一行占位(Token 压缩钩子使用)
_FOLD_PLACEHOLDER = "[更早的对话已折叠]"
# 滚动摘要后,最旧部分用的一行占位(60% 窗口滚动摘要使用)
_ROLLUP_PLACEHOLDER = "[更早的对话已滚动摘要存档]"

# 压缩时保留的最近对话轮数默认值(每轮 = 1 条用户 + 1 条顾问)
_DEFAULT_KEEP_ROUNDS = 8


class ConversationContext(BaseContext):
    """会话历史上下文:按 session_id 聚合用户/顾问消息。

    keep_rounds: 触发 Token 压缩时保留的最近对话轮数,更早的折叠成占位。
    oldest_rolled_up: 已滚动摘要的最旧消息条数(0 = 尚未滚动)。
                      滚动摘要后,渲染时这部分被替换为 _ROLLUP_PLACEHOLDER,
                      摘要正文已持久化到 long_term_memories(由记忆上下文召回)。
    """

    name = "conversation"
    priority = PRIORITY_CONVERSATION

    def __init__(self, keep_rounds: int = _DEFAULT_KEEP_ROUNDS, history_provider=None):
        self.chats_by_session_id: Dict[str, InMemoryChatMessageHistory] = {}
        self.keep_rounds = keep_rounds
        # 可注入的历史提供者:session_id -> InMemoryChatMessageHistory。
        # 注入后,本上下文读的是统一写入层(主控入口 + ConversationService)维护的
        # 同一份 session 历史,而不是各自私有副本 —— 避免"同条消息两次"或
        # "重复说的话被按 content 去重误删"。未注入(None)时沿用本地字典(向后兼容)。
        self.history_provider = history_provider
        # 滚动摘要水位:session_id -> 已折叠的消息条数(只增不减,天然防重复)
        self.rolled_up_count: Dict[str, int] = {}

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """取(必要时创建)某个 session 的历史对象。

        注入了 history_provider 时委托给它(读统一历史的同一对象)。
        """
        if self.history_provider is not None:
            return self.history_provider(session_id)
        chat_history = self.chats_by_session_id.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chats_by_session_id[session_id] = chat_history
        return chat_history

    def _render(self, messages) -> str:
        """把消息列表渲染成「用户/顾问: 内容」的多行文本。"""
        lines = []
        for msg in messages:
            role = "用户" if msg.type == "human" else "顾问"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _rolled_up_count(self, session_id: str) -> int:
        """某 session 已滚动摘要的消息条数。"""
        return self.rolled_up_count.get(session_id, 0)

    def mark_rolled_up(self, session_id: str, message_count: int) -> None:
        """记录某 session 最旧 message_count 条消息已滚动摘要(幂等,只增)。"""
        current = self._rolled_up_count(session_id)
        if message_count > current:
            self.rolled_up_count[session_id] = message_count

    def build(self, manager) -> str:
        """把当前会话历史完整渲染成「用户/顾问: 内容」的多行文本。

        已滚动摘要的最旧部分替换为一行占位,未滚动部分原样渲染。
        """
        session_id = getattr(manager, "session_id", None)
        if not session_id:
            return _NO_HISTORY

        chat_history = self.get_history(session_id)
        if not chat_history.messages:
            return _NO_HISTORY

        rolled = self._rolled_up_count(session_id)
        messages = chat_history.messages
        if rolled <= 0:
            return self._render(messages)

        return self.render_history(messages, rolled)

    def render_history(self, messages, rolled: int = 0, keep_rounds=None) -> str:
        """纯渲染已读取的消息快照;压缩不能重新带回已存档区间。"""
        remaining = list(messages)[rolled:]
        placeholders = [_ROLLUP_PLACEHOLDER] if rolled else []
        if keep_rounds is not None:
            keep = max(1, keep_rounds * 2)
            if len(remaining) > keep:
                remaining = remaining[-keep:]
                placeholders.append(_FOLD_PLACEHOLDER)
        body = self._render(remaining)
        if body:
            placeholders.append(body)
        return "\n".join(placeholders) or _NO_HISTORY

    def build_compressed(self, manager) -> str:
        """压缩渲染:仅保留最近 self.keep_rounds 轮,更早的折叠成一行占位。

        Token 超阈值时由 ContextManager._compress 调用,保证历史仍在
        却显著缩短,且最近内容(对当前追问最相关)完整保留。
        已滚动摘要的最旧部分仍用滚动占位(不重复出现折叠内容)。
        """
        session_id = getattr(manager, "session_id", None)
        if not session_id:
            return _NO_HISTORY

        chat_history = self.get_history(session_id)
        messages = list(chat_history.messages)
        if not messages:
            return _NO_HISTORY

        rolled = self._rolled_up_count(session_id)
        return self.render_history(messages, rolled, keep_rounds=self.keep_rounds)