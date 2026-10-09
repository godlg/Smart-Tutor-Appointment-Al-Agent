"""
响应生成器

负责生成AI响应内容
"""

import logging
from contextlib import aclosing
from typing import AsyncGenerator
from agents.context.types import ContextBudgetExceeded

logger = logging.getLogger(__name__)
from langchain_core.language_models.chat_models import BaseChatModel
from .prompt_builder import PromptBuilder


class ConsultationStreamError(Exception):
    """生成未完整成功；只由咨询处理器转换为安全的用户错误提示。"""


def visible_chunk_text(chunk) -> str:
    """只接受正文 content/text，不序列化思考、工具调用或元数据。"""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


class ResponseGenerator:
    """响应生成器（内置统一上下文管理，P0 试点）"""

    def __init__(self, llm: BaseChatModel, context_manager=None):
        self.llm = llm
        # 统一上下文管理器（可空：为空时回退到 prompt_builder 旧组装，保持向后兼容）
        self.context_manager = context_manager
        self.prompt_builder = PromptBuilder()

    def _build_prompt(self, user_input: str, knowledge_docs: list) -> str:
        """组装完整 prompt。

        - 未注入 context_manager 时,沿用原有 build_consultation_prompt(回归旧行为);
        - 注入后,用 ContextManager.build 产出完整输入(system + memory + tools +
          conversation + task),并追加 RAG 检索结果作为临时 knowledge 上下文。
        """
        if self.context_manager is None:
            return self.prompt_builder.build_consultation_prompt(user_input, knowledge_docs)

        knowledge_text = self.prompt_builder.build_knowledge_context(knowledge_docs)
        return self.context_manager.build(
            task=user_input,
            session_id=getattr(self.context_manager, "session_id", None),
            extra_contexts={"knowledge": knowledge_text},
            max_tokens=getattr(self.context_manager, "max_tokens", None),
        )

    async def generate_response(self, user_input: str, knowledge_docs: list) -> str:
        """生成标准响应"""
        try:
            prompt = self._build_prompt(user_input, knowledge_docs)
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content
        except ContextBudgetExceeded:
            return "问题或必需上下文过长，请缩短输入后重试。"
        except Exception:
            logger.exception("咨询响应生成失败")
            return "抱歉，处理您的问题时出现了错误。请稍后再试。"

    async def generate_response_stream(self, user_input: str, knowledge_docs: list) -> AsyncGenerator[str, None]:
        """预处理完成后直接透传模型正文块；关闭时同步关闭底层流。"""
        has_body = False
        try:
            prompt = self._build_prompt(user_input, knowledge_docs)
            async with aclosing(self.llm.astream([{"role": "user", "content": prompt}])) as stream:
                async for chunk in stream:
                    text = visible_chunk_text(chunk)
                    if not text:
                        continue
                    if not has_body:
                        has_body = True
                        yield "[REPLY][咨询机器人]"
                    yield text
            if not has_body:
                raise ConsultationStreamError("empty visible response")
        except ContextBudgetExceeded:
            # 预算拒绝保持旧协议，由处理器输出且不写成功行为。
            raise
        except Exception as exc:
            # 不记录上游原始异常/traceback，避免认证信息进入日志或用户输出。
            logger.warning("咨询模型流未完成: %s", type(exc).__name__)
            raise ConsultationStreamError("consultation generation failed") from None
    
    def create_unrelated_message(self) -> str:
        """创建与咨询无关的回复消息"""
        return "[THOUGHT][咨询机器人] 咨询机器人：这个问题不是咨询类问题，我将转回给归类机器人处理。"
