"""
正式课 Agent 入口（Phase 1：单次课 + 调课/退课）
"""

import uuid
from langchain_core.chat_history import InMemoryChatMessageHistory
from config.model_provider import create_chat_model
from .lesson_processor import LessonProcessor
from .change_processor import LessonChangeProcessor


class LessonAgent:
    """正式课 Agent 主控制器"""

    # 走调课/退课分支的意图（其余按正式课预约处理）
    CHANGE_INTENTS = ("reschedule", "cancel")

    def __init__(self, session_id=None, unrelated_callback=None, context_manager=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.unrelated_callback = unrelated_callback
        self.state = None
        # 统一上下文管理器(可空):注入后字段提取/调退课解析由它组装输入,否则回退旧逻辑
        self.context_manager = context_manager

        self.llm = create_chat_model(temperature=0)
        self.processor = LessonProcessor(self.llm, context_manager=self.context_manager)
        self._change_processor = None  # 惰性创建，只有真正调课/退课时才初始化

        self.chats_by_session_id = {}
        self.chat_history = self._get_chat_history(self.session_id)
        self.reset()

    def _build_context_manager(self):
        """组装正式课 Agent 用统一上下文管理器。

        价值点:conversation(多轮追问上下文)+ structured_output(JSON 提取约束)。
        返回的 manager 可注入 LessonAgent(context_manager=...)。
        """
        from agents.context import (
            ContextManager,
            SystemPromptContext,
            ConversationContext,
            StructuredOutputContext,
            ToolContext,
        )
        manager = ContextManager([
            SystemPromptContext(),
            ConversationContext(),
            StructuredOutputContext(),
            ToolContext(),
        ])
        manager.session_id = self.session_id
        return manager

    def _get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        chat_history = self.chats_by_session_id.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chats_by_session_id[session_id] = chat_history
        return chat_history

    def reset(self):
        self.lesson_history = {
            "subject": None,
            "grade_level": None,
            "start_time": None,
            "duration_minutes": "60",
            "preferred_teacher_name": None,
            "preferred_style": None,
            "package_id": None,
            # 递归防护:同一实例连续两次把输入判为"无关"踢回主路由时,主路由重分类
            # 仍命中本站 → 二次直接兜底不再调 callback,防止无限递归。
            "unrelated_escalations": 0,
        }
        self.finished = False
        self.chat_history.clear()

    def set_shared_state(self, shared_state):
        self.state = shared_state

    @property
    def change_processor(self) -> LessonChangeProcessor:
        """调课/退课处理器（首次使用时才创建，避免拖慢预约主流程的初始化）"""
        if self._change_processor is None:
            self._change_processor = LessonChangeProcessor(
                self.llm, context_manager=self.context_manager
            )
        return self._change_processor

    async def run_stream(self, user_input=None, intent="lesson"):
        if user_input is None:
            user_input = input("家长：")

        # 调课/退课走独立分支：复用 RescheduleHandler / CancelHandler，
        # 不能落到下面的预约流程（那会误当成「预约一节新课」）
        if intent in self.CHANGE_INTENTS:
            async for token in self.change_processor.handle_change_request(
                user_input, intent
            ):
                yield token
            self._reset_state_after_lesson()
            return

        ai_content = ""
        for token in self.processor.input_parser.parse_stream(user_input, self.chat_history):
            ai_content += token

        try:
            data = self.processor.input_parser.parse_data(ai_content)
            self.finished = self.processor.update_history_from_data(self.lesson_history, data)

            # 防御：家长已给出明确学科（含"代数"这类口语化别名，非空且非"未知"），
            # 即便 LLM 把 unrelated 误判为 true，也绝不应踢回主路由——否则主路由再分类
            # 又落到 lesson，死循环。这类半结构化、明确是预约意图但缺字段的输入，
            # 应落回 handle_incomplete_info 追问补齐学科/年级/时间。
            subject = data.get("subject")
            has_subject_signal = bool(subject and subject != "未知")

            if (
                data.get("unrelated", False)
                and not has_subject_signal
                and not self.lesson_history.get("awaiting_confirmation")
            ):
                if self.state:
                    from config.constants import StateEnum
                    self.state.value = StateEnum.CLASSIFY
                escalated = self.lesson_history.get("unrelated_escalations", 0)
                if escalated >= 1:
                    # 防御:同一输入已踢回过一次主路由(主路由重分类仍命中本次)。
                    # 二次命中 → 不再调 callback(避免无限递归),直接兜底并重置计数。
                    self.lesson_history["unrelated_escalations"] = 0
                    async for token in self.processor.handle_unrelated_request(
                        user_input, None, self.state
                    ):
                        yield token
                    return
                self.lesson_history["unrelated_escalations"] = 1
                async for token in self.processor.handle_unrelated_request(
                    user_input, self.unrelated_callback, self.state
                ):
                    yield token
                return

            if self.finished:
                async for token in self.processor.handle_complete_lesson(
                    self.lesson_history, self.session_id
                ):
                    yield token
                if not self.lesson_history.get("awaiting_confirmation"):
                    self._reset_state_after_lesson()
                return

            async for token in self.processor.handle_incomplete_info(data, self.lesson_history):
                yield token

        except Exception:
            yield self.processor.message_builder.create_parse_error_message()

    def _reset_state_after_lesson(self):
        self.reset()
        if self.state:
            from config.constants import StateEnum
            self.state.value = StateEnum.CLASSIFY