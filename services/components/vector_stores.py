"""
VectorStore 三类实现: dense (FAISS) / bm25 / hybrid。

全部包装现有服务, 不重写检索逻辑:
- dense: 复用 KnowledgeService 的向量检索 (search 接口 + embedding 落库)
- bm25: 复用 Bm25Index (jieba + BM25Okapi)
- hybrid: 复用 HybridRetriever (Dense + BM25 + RRF + 可选精排)

统一对外形态:
- add(docs): 把 docs 送入底层 (dense 落库并建索引; bm25 建内存索引;
  hybrid 两者皆需, 落库后由 HybridRetriever 在 search 时现建 BM25)。
- search(query, top_k): 返回 [{id, content, score, rank, ...}, ...] 降序。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .base import BaseVectorStore
from ..bm25_index import Bm25Index
from ..hybrid_retriever import HybridRetriever


def _run(coro):
    """同步接口包 async 实现 (search 是 async 的)。"""
    return asyncio.run(coro)


class DenseVectorStore(BaseVectorStore):
    """包装 KnowledgeService 的纯 Dense 向量检索。"""

    def __init__(self, knowledge_service: Any, **kwargs: Any) -> None:
        self._ks = knowledge_service

    def add(self, docs: List[Dict[str, Any]]) -> None:
        # knowledge_service 的 add_document 是 async; 逐条落库
        for doc in docs:
            _run(self._ks.add_document(
                content=doc["content"],
                category=doc.get("category", "general"),
                keywords=doc.get("keywords", []),
            ))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return _run(self._ks.search(query=query, top_k=top_k))


class Bm25VectorStore(BaseVectorStore):
    """包装 Bm25Index 的稀疏检索。"""

    def __init__(self, bm25_index: Any = None, **kwargs: Any) -> None:
        self._index = bm25_index if bm25_index is not None else Bm25Index()
        self._doc_map: Dict[int, Dict[str, Any]] = {}

    def add(self, docs: List[Dict[str, Any]]) -> None:
        self._index.build(docs)
        self._doc_map = {int(d["id"]): d for d in docs}

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        hits = self._index.search(query, top_n=top_k)
        results: List[Dict[str, Any]] = []
        for doc_id, rank, score in hits:
            doc = self._doc_map.get(doc_id)
            if doc is None:
                continue
            item = dict(doc)
            item["score"] = score
            item["rank"] = rank
            results.append(item)
        return results


class HybridVectorStore(BaseVectorStore):
    """包装 HybridRetriever (Dense + BM25 + RRF + 可选精排)。"""

    def __init__(
        self,
        knowledge_service: Any,
        dense_top_n: int = 20,
        bm25_top_n: int = 20,
        rrf_k: int = 60,
        rerank: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self._retriever = HybridRetriever(
            knowledge_service,
            rrf_k=rrf_k,
            candidates=max(dense_top_n, bm25_top_n),
            rerank=rerank,
        )
        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k

    def add(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            _run(self._retriever.ks.add_document(
                content=doc["content"],
                category=doc.get("category", "general"),
                keywords=doc.get("keywords", []),
            ))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return _run(self._retriever.search(query=query, top_k=top_k))
