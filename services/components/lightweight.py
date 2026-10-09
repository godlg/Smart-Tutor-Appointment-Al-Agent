"""
LLM / Embedding / Splitter 三类轻量组件实现。

- llm.openai_compatible: 包装 config.model_provider.create_chat_model
- embedding.openai_compatible: 包装 services.text_embedding.embed_input
- splitter.recursive: 包装 services.text_chunker.TextChunker
"""

from __future__ import annotations

from typing import Dict, List

from .base import BaseEmbedding, BaseLLM, BaseSplitter
from ..text_chunker import TextChunker
from ..text_embedding import embed_input
from config.model_provider import create_chat_model


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------

class OpenAICompatibleLLM(BaseLLM):
    """包装 create_chat_model() 的聊天模型 (env 切换 provider)。

    注: 不做真实 API 调用由调用方决定; chat() 直接透传给 langchain
    model.invoke(messages) 的 .content。
    """

    def __init__(self, temperature: float = 0.0) -> None:
        self._model = create_chat_model(temperature=temperature)
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        if temperature != self.temperature:
            self._model.temperature = temperature
            self.temperature = temperature
        result = self._model.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)


# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------

class OpenAICompatibleEmbedding(BaseEmbedding):
    """包装 embed_input() 的单文本向量接口 → 批量 embed()。"""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [embed_input(t) for t in texts]


# ----------------------------------------------------------------------
# Splitter
# ----------------------------------------------------------------------

class RecursiveSplitter(BaseSplitter):
    """包装 TextChunker (RecursiveCharacterTextSplitter)。

    空文本抛 EmptyTextError 的语义对外收敛为返回空列表(接口约定)。
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        from ..text_chunker import ChunkerSettings

        self._chunker = TextChunker(
            ChunkerSettings(chunk_size=chunk_size, chunk_overlap=overlap)
        )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        try:
            return [c.text for c in self._chunker.split(text)]
        except Exception:
            return []
