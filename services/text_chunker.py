"""
文本分块器 (P1-13)

封装 langchain-text-splitters 的 RecursiveCharacterTextSplitter,
让 service 层不必直接 import 第三方库。
immutable: TextChunk 与 ChunkerSettings 是 frozen dataclass。

设计要点:
- chunk_size / chunk_overlap 可配置,默认 500 / 50
  (与 config/components.yaml 的 splitter 参数一致; 500 中文字 ≈ 500 token < 512,
  精排 cross-encoder 不再截断 chunk 后半段)
- 中文友好:separators 包含 "。" "；" 等中文句末标点
- 空文本抛 EmptyTextError,让上层统一捕获
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmptyTextError(Exception):
    """输入文本为空 (或仅空白) 时抛出。"""
    pass


@dataclass(frozen=True)
class TextChunk:
    """一个分块,带 0-based 序号。
    frozen:创建后不可变,避免后续流程误改 text 内容。"""
    text: str
    chunk_index: int


@dataclass(frozen=True)
class ChunkerSettings:
    """分块器配置。默认值与 config/components.yaml 的 splitter 参数保持一致。"""
    chunk_size: int = 500
    chunk_overlap: int = 50


# 中英文混合友好的分隔符优先级:段落 > 行 > 句末标点 > 字
DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    " ",
    "",
]


class TextChunker:
    """文本 → List[TextChunk] 的薄包装。"""

    def __init__(self, settings: ChunkerSettings = ChunkerSettings()) -> None:
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> List[TextChunk]:
        """把 text 切成若干块,返回带序号的 TextChunk 列表。

        Args:
            text: 待分块文本

        Returns:
            至少 0 个 TextChunk;空文本抛 EmptyTextError

        Raises:
            EmptyTextError: text 为空 / 仅空白
        """
        if not text or not text.strip():
            raise EmptyTextError("text 不能为空")
        fragments = self._splitter.split_text(text)
        # 过滤 splitter 偶尔返回的空串(行尾 boundary 触发)
        return [
            TextChunk(text=frag, chunk_index=i)
            for i, frag in enumerate(fragments)
            if frag and frag.strip()
        ]