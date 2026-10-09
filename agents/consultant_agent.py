"""
咨询 Agent 主控制器（家教场景）
"""

import uuid
from contextlib import aclosing
from config.model_provider import create_chat_model
from .consultant import (
    KnowledgeRetriever,
    ConsultationClassifier,
    ResponseGenerator,
    ConsultationProcessor,
)
from .consultant.subject_extractor import default_profile_filters


class ConsultantAgent:
    """咨询 Agent"""

    def __init__(self, session_id=None, session_chat_history=None, identity=None,
                 conversation_service=None):
        if identity is not None and conversation_service is None and session_chat_history is not None:
            raise ValueError("身份咨询的共享历史要求注入可校验的会话存储")
        if identity is not None and session_id and identity.session_id and session_id != identity.session_id:
            raise ValueError("咨询 session 与已验证 identity 不一致")
        self.session_id = session_id or getattr(identity, "session_id", None) or str(uuid.uuid4())
        self.identity = identity
        self.conversation_service = conversation_service
        self.shared_state = None
        self.unrelated_callback = None
        # P0-1: 主控层维护的跨 Agent 共享会话历史(可空)。与 TrialAgent /
        # RecommendationAgent 同型:只读,用于抽取本轮检索的学科/年级硬约束。
        self.session_chat_history = session_chat_history

        self.llm = self._initialize_llm()

        # P0 试点:统一 5 类上下文管理器(system/conversation/memory/tools/structured)
        self.context_manager = self._build_context_manager()

        self.knowledge_retriever = KnowledgeRetriever()
        self.consultation_classifier = ConsultationClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm, context_manager=self.context_manager)
        self.consultation_processor = ConsultationProcessor(
            self.knowledge_retriever,
            self.consultation_classifier,
            self.response_generator,
            # 抽不到学科/年级时的最后兜底:默认学生画像
            profile_provider=(default_profile_filters if identity is None or identity.legacy
                              else lambda: (None, None)),
            identity=identity,
            behavior_service=(conversation_service.db_router.user_behavior_repo
                              if conversation_service is not None else None),
        )

    def _build_context_manager(self):
        """组装统一上下文管理器,并把会话 id 绑定,供会话历史等按会话聚合。"""
        if self.identity is not None and self.conversation_service is not None:
            from agents.context.runtime import build_runtime_context
            return build_runtime_context(self.identity, self.session_chat_history,
                                         self.conversation_service, include_memory=True)
        from agents.context import (
            ContextManager,
            SystemPromptContext,
            ConversationContext,
            MemoryContext,
            StructuredOutputContext,
            ToolContext,
        )
        provider = (lambda sid: self.session_chat_history) if self.session_chat_history is not None else None
        manager = ContextManager([
            SystemPromptContext(),
            ConversationContext(history_provider=provider),
            MemoryContext(identity=self.identity),
            StructuredOutputContext(),
            ToolContext(),
        ], identity=self.identity)
        manager.session_id = self.session_id
        return manager

    def _initialize_llm(self):
        return create_chat_model(temperature=0.3)

    async def __aenter__(self):
        await self.knowledge_retriever.initialize()
        print("咨询机器人已启动（家教模式）")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def set_shared_state(self, shared_state):
        self.shared_state = shared_state

    def set_unrelated_callback(self, callback):
        self.unrelated_callback = callback

    async def consult(self, user_input: str) -> str:
        return await self.consultation_processor.process_consultation(user_input)

    async def consult_stream(self, user_input: str):
        is_consultation = await self.consultation_classifier.is_consultation_related(user_input)

        if not is_consultation:
            async for token in self.consultation_processor.handle_unrelated_request(
                user_input, self.unrelated_callback, self.shared_state
            ):
                yield token
            return

        async with aclosing(self.consultation_processor.process_consultation_stream(
            user_input,
            self.session_id,
            query_context=self._build_rag_query_context(),
            history=getattr(self, "session_chat_history", None),
            memory_text=self._build_memory_profile_text(),
        )) as stream:
            async for token in stream:
                yield token

        self._reset_state_after_consultation()

    def _reset_state_after_consultation(self):
        from config.constants import FlowState
        if self.shared_state:
            self.shared_state.value = FlowState.CLASSIFY

    def _build_rag_query_context(self) -> str:
        """从统一上下文管理器中提取会话历史文本,供 RAG 检索做语义召回增强。

        只取有界最近历史与显式近期摘要,不把 system/tools 拼进 embedding。
        首轮/无历史且无摘要时返回空串,检索回退为纯当前输入。
        """
        return self.context_manager.build_retrieval_context()

    def _build_memory_profile_text(self) -> str:
        """取用户画像摘要文本(仅画像,不做长期记忆向量召回)。

        供 P0-1 学科抽取读取画像里的学科偏好;画像不可用时返回空串,
        抽取逻辑继续回退到默认学生画像,不阻断检索。
        """
        from agents.context.contexts.memory import MemoryContext
        # Validate before the optional-profile try: mixed identity is not an
        # unavailable optional profile, and must not trigger any storage read.
        self.context_manager.validate_identity_contexts()
        mem = next(
            (c for c in self.context_manager.contexts if isinstance(c, MemoryContext)),
            None,
        )
        if mem is None:
            return ""
        try:
            return mem.build_profile_summary() or ""
        except Exception:
            return ""