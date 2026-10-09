"""
无关请求处理器（家教场景）
"""

from typing import AsyncGenerator
from .state_manager import StateManager


class UnrelatedHandler:
    """无关请求处理器"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._default_replies = [
            "抱歉，我无法处理这个问题。我只能帮您处理 K12 家教相关的咨询和预约。请问您想了解课程、预约试讲，还是选购套餐呢？",
            "很抱歉，我专门负责家教相关的服务。如果您想了解课程体系或进行预约，我很乐意为您提供帮助！",
            "对不起，这个问题超出了我的服务范围。我主要协助处理课程咨询、试讲预约和排课，有什么可以帮您的吗？",
        ]
        self._reply_index = 0

    async def handle_unrelated_sync(self, user_input: str) -> str:
        self.state_manager.reset_to_classify()
        return self._get_next_reply()

    async def handle_unrelated_async(self, user_input: str) -> AsyncGenerator[str, None]:
        self.state_manager.reset_to_classify()
        reply = self._get_next_reply()
        yield "[REPLY][归类机器人]"
        for char in reply:
            yield char

    def _get_next_reply(self) -> str:
        reply = self._default_replies[self._reply_index]
        self._reply_index = (self._reply_index + 1) % len(self._default_replies)
        return reply

    def add_custom_reply(self, reply: str) -> None:
        if reply and reply not in self._default_replies:
            self._default_replies.append(reply)

    def set_business_context(self, service_name: str = "家教课程") -> None:
        self._default_replies = [
            f"抱歉，我无法处理这个问题。我只能帮您处理{service_name}相关的咨询和预约。请问您需要了解课程、预约试讲，还是选购套餐？",
            f"很抱歉，我专门负责{service_name}相关的服务。如果您想了解课程内容或进行预约，我很乐意为您提供帮助！",
            f"对不起，这个问题超出了我的服务范围。我主要协助处理{service_name}咨询、试讲预约和排课，有什么可以为您服务的吗？",
        ]

    def get_get_available_replies(self) -> list:
        return self._default_replies.copy()

    def reset_reply_rotation(self) -> None:
        self._reply_index = 0