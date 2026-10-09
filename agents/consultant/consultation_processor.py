"""
咨询流程处理器

负责协调整个咨询流程
"""

import logging
from contextlib import aclosing
from typing import AsyncGenerator, Callable, Dict, Any, Optional, Tuple
from agents.context.types import ContextBudgetExceeded

logger = logging.getLogger(__name__)
from .knowledge_retriever import KnowledgeRetriever
from .consultation_classifier import ConsultationClassifier
from .response_generator import ResponseGenerator
from .subject_extractor import resolve_subject_grade


class ConsultationProcessor:
    """咨询流程处理器"""

    def __init__(self, knowledge_retriever: KnowledgeRetriever,
                 consultation_classifier: ConsultationClassifier,
                 response_generator: ResponseGenerator,
                 profile_provider: Optional[
                     Callable[[], Tuple[Optional[str], Optional[str]]]
                 ] = None, identity=None, behavior_service=None):
        self.knowledge_retriever = knowledge_retriever
        self.consultation_classifier = consultation_classifier
        self.response_generator = response_generator
        # P0-1: 抽取学科/年级时的最后兜底(默认学生画像)。
        # 未注入(None)则不兜底 —— 直接构造 processor 的单元测试因此保持确定性,
        # 真实聊天链路由 ConsultantAgent 注入 default_profile_filters。
        self.profile_provider = profile_provider
        self.identity = identity
        self.behavior_service = behavior_service

    async def process_consultation(self, user_input: str) -> str:
        """处理标准咨询"""
        # 1. 检索知识
        knowledge_docs = await self.knowledge_retriever.search_knowledge(user_input, top_k=3)
        
        # 2. 生成响应
        response = await self.response_generator.generate_response(user_input, knowledge_docs)
        
        return response
    
    async def process_consultation_stream(
        self,
        user_input: str,
        session_id: str,
        query_context: str = "",
        history=None,
        memory_text: str = "",
    ) -> AsyncGenerator[str, None]:
        """处理流式咨询

        query_context: 来自 ConversationContext 的历史摘要文本,拼进 RAG 检索,
        让指代不清的追问(如"计算薄弱")能借历史上下文(初二、数学)召回更准。
        history: 本 session 的 InMemoryChatMessageHistory(可空),供 P0-1 抽学科/年级。
        memory_text: 用户画像摘要文本(可空),作为学科偏好的兜底来源。

        P0-1: 检索前先用规则从「当前输入 + 最近消息 + 上下文摘要 + 画像」抽出
        subject / grade_level,透传给 search_knowledge,让 filter-then-vector 真正生效。
        """
        has_reply = False
        try:
            # 1. 抽学科 / 年级硬约束(规则,不新增 LLM 调用)
            subject, grade_level = resolve_subject_grade(
                user_input,
                history=history,
                context_texts=(query_context, memory_text),
                profile_provider=self.profile_provider,
            )

            # 2. 检索知识(带上对话上下文增强语义召回 + 学科/年级硬约束)
            knowledge_docs = await self.knowledge_retriever.search_knowledge(
                user_input, top_k=3, query_context=query_context,
                subject=subject, grade_level=grade_level,
            )

            # 3. 生成响应
            async with aclosing(self.response_generator.generate_response_stream(
                user_input, knowledge_docs
            )) as stream:
                async for token in stream:
                    has_reply = has_reply or "[REPLY]" in token
                    yield token

            # 4. 记录用户行为
            await self._record_consultation_behavior(user_input, knowledge_docs, session_id)
            
        except ContextBudgetExceeded:
            yield "[REPLY][咨询机器人]问题或必需上下文过长，请缩短输入后重试。"
        except Exception as exc:
            logger.warning("咨询流程未完成: %s", type(exc).__name__)
            prefix = "" if has_reply else "[REPLY][咨询机器人]"
            yield prefix + "[ERROR]抱歉，咨询回答未能完成，请稍后重试。"
    
    async def handle_unrelated_request(self, user_input: str, unrelated_callback, shared_state) -> AsyncGenerator[str, None]:
        """处理与咨询无关的请求"""
        # 重置状态
        if shared_state:
            from config.constants import FlowState
            shared_state.value = FlowState.CLASSIFY

        yield self.response_generator.create_unrelated_message()

        # 转给回调处理
        if unrelated_callback:
            async for token in unrelated_callback(user_input):
                yield token
    
    async def _record_consultation_behavior(self, user_input: str, knowledge_docs: list, session_id: str):
        """记录咨询行为"""
        try:
            action_data = {
                'question': user_input,
                'knowledge_docs_used': len(knowledge_docs),
                'categories': list(set(doc.get('category', 'unknown') for doc in knowledge_docs)) if knowledge_docs else []
            }
            
            if self.identity is not None:
                # 匿名student namespace只用于本身份的咨询行为,不是业务学员授权。
                service = self.behavior_service
                if service is None:
                    logger.warning("咨询行为未写入：身份缺少注入的共享行为服务")
                    return
                result = service.record_behavior(
                    user_id=self.identity.memory_scope(), action_type='consultation',
                    action_data=action_data, session_id=self.identity.session_id or session_id,
                )
            else:
                from agents.user_behavior_agent import UserBehaviorAgent
                behavior_agent = UserBehaviorAgent()
                result = behavior_agent.record_behavior(
                    action_type='consultation', action_data=action_data, session_id=session_id,
                )
            if not result:
                logger.warning("咨询行为持久化失败")

        except Exception:
            logger.exception("记录咨询行为失败")
