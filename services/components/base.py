"""
六类组件抽象基类 (ABC)。

设计约束:
- 每类只定义"最少方法", 让现有实现可被薄包装接入。
- VectorStore 统一返回 List[Dict] (与 knowledge_service.search 结构一致,
  含 content / score / rank 字段), 这样 Evaluator 可按 doc_id 评测。
- Evaluator 接口为 EDD (Experiment-Driven Development) 预留: 输入一个
  retriever 对象, 输出指标字典。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


class BaseLLM(ABC):
    """LLM 聊天组件抽象。"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """messages: [{"role": "user"/"system"/"assistant", "content": ...}], 返回生成文本。"""


class BaseEmbedding(ABC):
    """文本向量化组件抽象。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量计算文本向量, 返回与输入等长的向量列表。"""


class BaseSplitter(ABC):
    """文本分块组件抽象。"""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """把长文本切成若干块, 返回块文本列表(不抛空输入异常, 空输入返回空列表)。"""


class BaseVectorStore(ABC):
    """向量库组件抽象 (Dense / BM25 / Hybrid 三选一)。"""

    @abstractmethod
    def add(self, docs: List[Dict[str, Any]]) -> None:
        """把 docs (含 id / content / embedding 等) 加入索引。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索 query, 返回按相关度降序的 docs (每项含 score / rank / id / content)。"""


class BaseReranker(ABC):
    """精排组件抽象 (可无操作: type=none 原样返回)。"""

    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """对候选 docs 重排, 返回按精排分降序的前 top_k 条(不修改原 dict)。"""


class BaseEvaluator(ABC):
    """检索评估器抽象 (EDD 地基: 对一个 retriever 输出指标)。"""

    @abstractmethod
    def evaluate(self, retriever: Any) -> Dict[str, Any]:
        """对一个具备 search(query, top_k) 方法的 retriever 评测, 返回指标字典。"""


@dataclass(frozen=True)
class QueryCase:
    """一个评测用例: 查询 + 期望命中的 doc_id 集合。

    供 BaseEvaluator 实现 (如 recall_at_k) 使用:
    每个 query 的检索结果与 expected_doc_ids 求交, 汇总出 Recall@k / MRR。
    """
    query: str
    expected_doc_ids: List[int] = field(default_factory=list)
