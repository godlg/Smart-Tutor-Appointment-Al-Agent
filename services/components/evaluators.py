"""
Reranker 实现: bge (包装 reranker.rerank) / none (原样返回)。

Evaluator 实现: recall_at_k (Recall@k + MRR, EDD 地基)。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseEvaluator, BaseReranker, QueryCase
from .. import reranker as reranker_module


# ----------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------

class BgeReranker(BaseReranker):
    """包装 services.reranker.rerank (bge-reranker / ms-marco, ONNX)。

    模型缺失时 available() 为 False, 此处原样返回 docs 的前 top_k 条,
    与现有 HybridRetriever 的静默回退行为保持一致。
    """

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not reranker_module.available():
            return docs[:top_k]
        return reranker_module.rerank(query, docs, top_k=top_k)


class NoOpReranker(BaseReranker):
    """type=none: 不精排, 原样返回前 top_k 条。"""

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return docs[:top_k]


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------

class RecallAtKEvaluator(BaseEvaluator):
    """召回命中率评测器: 对具备 search(query, top_k) 的 retriever 计算 Recall@k / MRR。

    输入: QueryCase 列表 (query + 期望命中的 doc_id 集合)。
    输出: {
        recall_at_k: float,   # 各 case Recall@k 的均值 (0..1)
        mrr: float,           # 各 case MRR 的均值 (0..1)
        case_results: [       # 每条 case 的明细, 便于调试
            {query, expected, hit_ids, recall, mrr}
        ],
    }
    注: 这是 EDD (Experiment-Driven Development) 的第一步地基:
    evaluate(retriever) 的接口形态即评估基线, 后续可扩展 precision@k / ndcg@k。
    """

    def __init__(self, cases: List[QueryCase], k: int = 3) -> None:
        self._cases = cases
        self.k = k

    def evaluate(self, retriever: Any) -> Dict[str, Any]:
        case_results: List[Dict[str, Any]] = []

        for case in self._cases:
            if not case.expected_doc_ids:
                continue
            results = retriever.search(case.query, top_k=self.k)
            expected = set(case.expected_doc_ids)
            hit_ids = [
                int(r.get("id"))
                for r in results
                if int(r.get("id")) in expected
            ]
            recall = len(hit_ids) / len(expected)
            # MRR: 首个命中位置 (rank 从 1 开始)
            first_rank = None
            for rank, r in enumerate(results, start=1):
                if int(r.get("id")) in expected:
                    first_rank = rank
                    break
            mrr = 1.0 / first_rank if first_rank is not None else 0.0

            case_results.append({
                "query": case.query,
                "expected": sorted(expected),
                "hit_ids": hit_ids,
                "recall": recall,
                "mrr": mrr,
            })

        n = len(case_results)
        # recall_at_k / mrr 取各 case 均值 (与 MRR 同口径)
        return {
            "recall_at_k": sum(c["recall"] for c in case_results) / n if n else 0.0,
            "mrr": sum(c["mrr"] for c in case_results) / n if n else 0.0,
            "case_results": case_results,
        }
