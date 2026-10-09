"""
任务分类 Agent 主控制器（家教场景）
"""

import logging
from contextlib import aclosing

from dotenv import load_dotenv
from config.model_provider import create_chat_model
from .task_classification import (
    TaskClassifier,
    StateManager,
    AgentRouter,
    UnrelatedHandler,
    ClassificationProcessor,
)

load_dotenv()

logger = logging.getLogger(__name__)


class TaskClassificationAgent:
    """任务分类主控制器"""

    def __init__(
        self,
        consultant_agent=None,
        trial_agent=None,
        lesson_agent=None,
        recommendation_agent=None,
        context_manager=None,
        session_id: str = None,
        handoff_service=None,
        session_chat_history=None,
    ):
        self.consultant_agent = consultant_agent
        self.trial_agent = trial_agent
        self.lesson_agent = lesson_agent
        self.recommendation_agent = recommendation_agent
        # 统一上下文管理器(可空):注入后主控分类由它组装输入,否则回退旧逻辑
        self.context_manager = context_manager

        # 人工接管:会话隔离的兜底状态(多次踢回自动转人工 / 主动转人工)
        self.session_id = session_id or "default"
        # 跨 Agent 共享的会话历史(可空,向后兼容);转人工时快照进 DB
        self.session_chat_history = session_chat_history
        if handoff_service is None:
            from services.human_handoff_service import HumanHandoffService
            handoff_service = HumanHandoffService()
        self.handoff_service = handoff_service

        self.llm = create_chat_model(temperature=0)

        self.state_manager = StateManager()
        self.task_classifier = TaskClassifier(
            self.llm, context_manager=self.context_manager, session_id=self.session_id
        )
        self.agent_router = AgentRouter(
            consultant_agent=consultant_agent,
            state_manager=self.state_manager,
            trial_agent=trial_agent,
            lesson_agent=lesson_agent,
            recommendation_agent=recommendation_agent,
        )
        self.unrelated_handler = UnrelatedHandler(self.state_manager)
        self.classification_processor = ClassificationProcessor(
            self.task_classifier,
            self.state_manager,
            self.agent_router,
            self.unrelated_handler,
            handoff_service=self.handoff_service,
            session_id=self.session_id,
            session_chat_history=self.session_chat_history,
        )

        self._setup_callbacks()
        self.state = self.state_manager.state

    def _snapshot_transcript(self):
        """安全地快照会话历史(history 为 None / 无 messages 时返回 None)。"""
        history = getattr(self, "session_chat_history", None)
        messages = getattr(history, "messages", None)
        return self.handoff_service.snapshot_transcript(messages)

    def _build_transcript(self):
        """会话历史快照 + 最近用户输入(无关轮兜底);history 为空也可用。"""
        history = getattr(self, "session_chat_history", None)
        messages = getattr(history, "messages", None)
        return self.handoff_service.build_transcript(self.session_id, messages)

    def _build_context_manager(self):
        """组装主控分类用统一上下文管理器。

        主控价值点:conversation(多轮上下文)+ structured_output(分类约束)。
        返回的 manager 可注入 TaskClassificationAgent(context_manager=...),
        供 TaskClassifier 分类时由它组装输入。

        注意:当前**未被 `__init__` 调用**,生产路径由 `api/chat_handler.py`
        注入 `build_runtime_context(...)`;本方法为预留工厂。
        """
        from agents.context import (
            ContextManager,
            SystemPromptContext,
            ConversationContext,
            StructuredOutputContext,
            ToolContext,
        )
        history = getattr(self, "session_chat_history", None)
        manager = ContextManager([
            SystemPromptContext(),
            ConversationContext(history_provider=(lambda sid: history) if history is not None else None),
            StructuredOutputContext(),
            ToolContext(),
        ])
        manager.session_id = self.session_id
        return manager

    def _setup_callbacks(self):
        if self.trial_agent and hasattr(self.trial_agent, "unrelated_callback"):
            self.trial_agent.unrelated_callback = self.handle_unrelated
        if self.lesson_agent and hasattr(self.lesson_agent, "unrelated_callback"):
            self.lesson_agent.unrelated_callback = self.handle_unrelated

    async def classify_task(self, task):
        # 记录用户输入(含无关轮),供转人工时补进 transcript 快照
        self.handoff_service.record_input(self.session_id, task)
        # 人工接管短路:已转人工 → 直接回短路文案,不进分类流程
        if self.handoff_service.is_handed_off(self.session_id):
            # 家长消息落库(供客服查看上下文);失败静默,不阻塞回复
            try:
                self.handoff_service.parent_message(self.session_id, task)
            except Exception as e:
                logger.warning(f"记录家长消息失败(已忽略): {e}")
            return self.handoff_service.handed_off_reply()
        # 主动转人工关键词:命中 → 触发转人工并回触发文案
        kw = self.handoff_service.match_keyword(task)
        if kw:
            self.handoff_service.trigger_handoff(
                self.session_id, "keyword", reason=kw,
                transcript=self._build_transcript(),
            )
            return self.handoff_service.handoff_reply()
        return await self.classification_processor.process_task_sync(task)

    async def classify_task_stream(self, task):
        # 记录用户输入(含无关轮),供转人工时补进 transcript 快照
        self.handoff_service.record_input(self.session_id, task)
        # 人工接管短路:已转人工 → 逐字回短路文案,不进分类流程
        if self.handoff_service.is_handed_off(self.session_id):
            # 家长消息落库(供客服查看上下文);失败静默,不阻塞回复
            try:
                self.handoff_service.parent_message(self.session_id, task)
            except Exception as e:
                logger.warning(f"记录家长消息失败(已忽略): {e}")
            yield "[REPLY]"
            for char in self.handoff_service.handed_off_reply():
                yield char
            return
        # 主动转人工关键词:命中 → 触发转人工并逐字回触发文案
        kw = self.handoff_service.match_keyword(task)
        if kw:
            self.handoff_service.trigger_handoff(
                self.session_id, "keyword", reason=kw,
                transcript=self._build_transcript(),
            )
            yield "[REPLY][HANDOFF]"
            for char in self.handoff_service.handoff_reply():
                yield char
            return
        async with aclosing(self.classification_processor.process_task_stream(task)) as stream:
            async for token in stream:
                yield token

    async def handle_unrelated(self, user_input):
        result = ""
        async for token in self.classification_processor.process_task_stream(user_input):
            result += token
        return result

    def get_classification_info(self):
        return self.classification_processor.get_current_state_info()

    def reset_conversation(self):
        self.classification_processor.reset_conversation()

    def set_business_context(self, service_name: str = None):
        if service_name is None:
            service_name = "家教课程"
        self.unrelated_handler.set_business_context(service_name)