# services/bm25_index.py
"""BM25 稀疏检索索引（中文分词）。

与 FAISS Dense 路互补:BM25 打捞精确词/专有名词(学科术语、课程名、政策名),
Dense 兜语义近义。两者排名经 RRF 融合。

- 分词: jieba.cut + 轻量停用词过滤, 保留英文/数字 token。
- 索引: 全内存 BM25Okapi, 知识库规模小, 每次 build 成本可接受。
- 对齐: build 传入的 docs 需与 Dense 路候选集一致(同一批 doc_id),
  RRF 融合按 doc_id 对齐。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

# 轻量停用词: 仅过滤高频虚词/口语词, 不做过强过滤(专业名词优先保留)
_STOPWORDS = frozenset(
    "的 了 和 是 在 有 我 你 他 她 它 这 那 与 及 或 等 个 种 吗 呢 吧 啊 "
    "请 问 怎么 如何 什么 为什么 哪些 哪个 一下 一下 可以 应该 老师 孩子 家长 我们 你们 他们".split()
)

_TOKEN_RE = re.compile(r"^[a-zA-Z0-9\u4e00-\u9fff]+$")


def tokenize(text: str) -> List[str]:
    """jieba 分词 + 停用词过滤 + 只保留中英数 token。

    >>> tokenize("初二数学 分数运算 薄弱")
    ['初二', '数学', '分数', '运算', '薄弱']
    """
    tokens = []
    for word in jieba.cut(text):
        word = word.strip().lower()
        if not word:
            continue
        if word in _STOPWORDS:
            continue
        if not _TOKEN_RE.match(word):
            continue
        tokens.append(word)
    return tokens


class Bm25Index:
    """基于 rank_bm25 的稀疏检索器。"""

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[int] = []
        self._corpus: List[str] = []
        self._built = False

    @property
    def built(self) -> bool:
        return self._built

    def build(self, docs: List[Dict[str, Any]]) -> "Bm25Index":
        """从候选文档构建索引。

        docs: list[dict], 至少含 id 与 content(可含 keywords, 拼入正文增强召回)。
        """
        self._doc_ids = []
        self._corpus = []
        for doc in docs:
            doc_id = doc.get("id")
            content = doc.get("content") or ""
            keywords = doc.get("keywords") or []
            if doc_id is None:
                continue
            text = f"{content} {' '.join(keywords)}" if keywords else content
            self._doc_ids.append(int(doc_id))
            self._corpus.append(text)
        tokenized = [tokenize(t) for t in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._built = True
        if self._bm25 is not None:
            self._smooth_zero_idf()
        return self

    def _smooth_zero_idf(self) -> None:
        """小型语料退化防护: 词出现在恰好/超过半数文档时 idf<=0, BM25 分数恒为 0。

        rank_bm25 只对负 idf 应用 epsilon 地板; idf 恰为 0(如 2 文档语料里
        每词都在一半文档出现)时分数全 0, 检索完全失效。这里给 idf<=0 的词
        一个最小正地板, 保持稀疏路至少能按词频排序。
        """
        floor = 0.1
        for word in self._bm25.idf:
            if self._bm25.idf[word] <= 0:
                self._bm25.idf[word] = floor

    def search(self, query: str, top_n: int = 20) -> List[Tuple[int, int, float]]:
        """检索, 返回 [(doc_id, rank, bm25_score)] 按分数降序。

        只返回有词命中的结果(score > 0), rank 从 1 开始。
        """
        if not self._built or self._bm25 is None or not self._doc_ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        scored: List[Tuple[int, float]] = []
        for doc_id, score in zip(self._doc_ids, scores):
            if score > 0:
                scored.append((doc_id, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc_id, rank, score)
            for rank, (doc_id, score) in enumerate(scored[:top_n], start=1)
        ]
