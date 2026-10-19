"""检索指标 (纯函数, 可单测)。

只放指标计算, 不依赖检索实现 / 网络 / LLM:
- ndcg_at_k: 归一化折损累计增益 @k (二元相关性)

Recall@k / MRR 仍由 services.components.evaluators.RecallAtKEvaluator 提供
(eval/runner 复用, 不在此重复实现), 本模块只补它没有的 NDCG。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Set


def _expected_set(expected_ids: Iterable[int]) -> Set[int]:
    return {int(x) for x in expected_ids}


def dcg_at_k(ranked_ids: Sequence[int], expected_ids: Iterable[int], k: int) -> float:
    """折损累计增益 DCG@k (二元相关性: 命中记 1, 否则 0)。

    DCG = Σ_{i=1..k} rel_i / log2(i + 1), i 为 1-based 排名。
    """
    expected = _expected_set(expected_ids)
    total = 0.0
    for i, doc_id in enumerate(list(ranked_ids)[:max(0, k)], start=1):
        if int(doc_id) in expected:
            total += 1.0 / math.log2(i + 1)
    return total


def ideal_dcg_at_k(expected_ids: Iterable[int], k: int) -> float:
    """理想 DCG@k: 期望文档全部排在最前面时的 DCG。"""
    n = min(len(_expected_set(expected_ids)), max(0, k))
    return sum(1.0 / math.log2(i + 1) for i in range(1, n + 1))


def ndcg_at_k(ranked_ids: Sequence[int], expected_ids: Iterable[int], k: int) -> float:
    """NDCG@k = DCG@k / IDCG@k。

    无可命中文档(|expected| == 0)时返回 0.0 (调用方应过滤这类 case);
    命中排名越靠前越接近 1.0。
    """
    ideal = ideal_dcg_at_k(expected_ids, k)
    if ideal <= 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, expected_ids, k) / ideal


def mean(values: Sequence[float]) -> float:
    """均值; 空序列返回 0.0。"""
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def empty_recall_rate(case_results: Sequence[Dict[str, Any]]) -> float:
    """空召回率: recall == 0 的 case 占比 (retrieval 整体退化最直观的指标)。"""
    if not case_results:
        return 0.0
    empty = sum(1 for c in case_results if float(c.get("recall", 0.0)) == 0.0)
    return empty / float(len(case_results))
