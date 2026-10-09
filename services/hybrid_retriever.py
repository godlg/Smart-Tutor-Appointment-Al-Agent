# services/hybrid_retriever.py
"""混合检索编排器: Dense(FAISS) + BM25 → RRF 融合 → ms-marco 精排。

检索五阶段(Query Trace 埋点):
    1. 候选集(全量 或 filter-then-vector 的 SQL 过滤子集)
    2. Dense 路: FAISS 内积检索
    3. BM25 路: jieba 分词 + BM25Okapi
    4. RRF 融合: score = Σ 1/(k + rank)
    5. 精排: ms-marco cross-encoder 打分(可开关; 失败自动回退 RRF 结果)

与 KnowledgeService.search()(纯 Dense) 完全解耦: 不修改原检索路径,
由调用方(knowledge_retriever)按 RETRIEVAL_MODE 选择 dense / hybrid。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .bm25_index import Bm25Index
from . import reranker
from .text_embedding import embed_input
from observability import trace_span

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def rerank_enabled_default() -> bool:
    """精排开关默认开(模型已随仓库 cache/ 提供)。"""
    return _env_bool("RERANK_ENABLED", True)


def rrf_merge(
    ranked_lists: List[List[Tuple[int, int]]], k: int = 60
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion。

    ranked_lists: 各路结果 [(doc_id, rank)], rank 从 1 开始;
    返回 [(doc_id, rrf_score)] 按分数降序。k 常用 60。
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """双路召回 + RRF + 精排。依赖注入 KnowledgeService(便于测试)。"""

    def __init__(
        self,
        knowledge_service: Any,
        rrf_k: int = 60,
        candidates: int = 20,
        rerank: Optional[bool] = None,
    ) -> None:
        self.ks = knowledge_service
        self.rrf_k = rrf_k
        self.candidates = candidates
        # None → 走环境变量默认(生产 hybrid 全开); 测试可显式关
        self.rerank = rerank if rerank is not None else rerank_enabled_default()

    async def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """混合检索, 返回结构与 KnowledgeService.search 一致。"""
        with trace_span(
            "rag_search",
            mode="hybrid",
            query=query[:30],
            top_k=top_k,
            has_meta_filter=any(v is not None for v in (subject, grade_level, doc_type)),
        ) as span:
            return await self._search_impl(
                query=query, top_k=top_k, category=category,
                subject=subject, grade_level=grade_level, doc_type=doc_type,
                span=span,
            )

    async def _search_impl(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        subject: Optional[str],
        grade_level: Optional[str],
        doc_type: Optional[str],
        span=None,
    ) -> List[Dict[str, Any]]:
        """混合检索主体(与 search 分离以便 span attrs 在结束时统一挂载)。"""
        stages: Dict[str, float] = {}
        t0 = time.perf_counter()

        # ---- 阶段1: 候选集(filter-then-vector 的 SQL 过滤在此发生) ----
        has_meta_filter = any(
            v is not None for v in (subject, grade_level, doc_type)
        )
        if has_meta_filter:
            candidates = self.ks.db.search_documents_with_filters(
                subject=subject, grade_level=grade_level, doc_type=doc_type,
            )
        else:
            candidates = self.ks.db.get_all_documents()
        # Dense/BM25 两路必须同一候选集: 只保留有 embedding 的(与 FAISS 索引对齐)
        candidates = [
            c for c in candidates
            if c.get("embedding") and c.get("content")
        ]
        if not candidates:
            return []
        doc_map: Dict[int, Dict[str, Any]] = {int(c["id"]): c for c in candidates}
        stages["candidates"] = time.perf_counter() - t0

        # ---- 阶段2: Dense 路(FAISS 内积, 与现有 IndexFlatIP 一致) ----
        t1 = time.perf_counter()
        dense_ranked = self._dense_search(query, candidates)
        stages["dense"] = time.perf_counter() - t1

        # ---- 阶段3: BM25 路 ----
        t2 = time.perf_counter()
        bm25 = Bm25Index().build(candidates)
        bm25_ranked = [
            (doc_id, rank)
            for doc_id, rank, _ in bm25.search(query, self.candidates)
        ]
        stages["bm25"] = time.perf_counter() - t2

        # ---- 阶段4: RRF 融合 ----
        t3 = time.perf_counter()
        merged = rrf_merge([dense_ranked, bm25_ranked], self.rrf_k)[
            : self.candidates
        ]
        merged_docs: List[Dict[str, Any]] = []
        for doc_id, rrf_score in merged:
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            item = dict(doc)
            item["rrf_score"] = rrf_score
            merged_docs.append(item)
        stages["rrf"] = time.perf_counter() - t3

        # ---- 阶段5: 精排(可开关; 失败回退 RRF) ----
        t4 = time.perf_counter()
        reranked: Optional[List[Dict[str, Any]]] = None
        if self.rerank and reranker.available():
            try:
                reranked = reranker.rerank(query, merged_docs, top_k=top_k)
            except Exception as e:
                logger.warning(f"精排失败, 回退 RRF 结果: {e}")
                reranked = None
        if reranked is None:
            reranked = merged_docs[:top_k]
        stages["rerank"] = time.perf_counter() - t4

        # ---- 结果封装(score = 精排分, 否则 rrf 融合分) ----
        results: List[Dict[str, Any]] = []
        for rank, doc in enumerate(reranked, start=1):
            if category and doc.get("category") != category:
                continue
            item = dict(doc)
            item["score"] = doc.get("rerank_score", doc.get("rrf_score", 0.0))
            item["rank"] = rank
            results.append(item)

        logger.info(
            "hybrid_search query=%r stages=%s took=%.1fms top=%d/%d",
            query[:30], stages, (time.perf_counter() - t0) * 1000,
            len(results), len(candidates),
        )
        # Trace 埋点:把五阶段耗时挂到 rag_search span attrs 上
        if span is not None:
            span.attrs["stages_ms"] = {
                k: round(v * 1000, 2) for k, v in stages.items()
            }
            span.attrs["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            span.attrs["result_count"] = len(results)
        return results

    def _dense_search(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """候选集上 FAISS 检索, 返回 [(doc_id, rank)]。"""
        embs = np.array(
            [c["embedding"] for c in candidates], dtype="float32"
        )
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        query_emb = np.array([embed_input(query)], dtype="float32")
        k = min(self.candidates, len(candidates))
        scores, indices = index.search(query_emb, k)
        ranked: List[Tuple[int, int]] = []
        for rank, idx in enumerate(indices[0], start=1):
            if 0 <= idx < len(candidates):
                ranked.append((int(candidates[idx]["id"]), rank))
        return ranked
