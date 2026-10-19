"""RAG 评估集质量守卫: 反逐字改写 (anti-verbatim)。

背景 (spec §0): v1 评估集里 query 基本是语料原文的逐字改写
(语料「高中数学 12 周提分方案…」→ query「高中数学 12 周提分方案包含哪些模块？」),
依赖 BM25 词法直命中, `Recall@3` 恒为 1.0, 门禁毫无区分度。

硬性规则: **query 与冻结 corpus 任一篇的最长公共子串必须 < MIN_COMMON_SUBSTRING**。
把原文抄进 query 会被这里拦下, 逼着 query 用「用户真实提问的说法」写。

纯函数, 不依赖检索实现 / 网络 / LLM, 建集脚本与测试共用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# 允许的最长公共子串上限 (字符数): 公共子串 >= 12 即判定为「抄原文」
MIN_COMMON_SUBSTRING = 12


def _has_common_substring_at_least(a: str, b: str, length: int) -> bool:
    """a 与 b 是否存在长度 >= length 的公共子串 (按 length 长窗口判断)。

    原理: 若存在长度 >= L 的公共子串, 它必然包含一个长度为 L 的公共窗口;
    反之若存在公共 L 窗口, 则公共子串长度 >= L。故 L 窗口判断与
    「最长公共子串 >= L」等价, 但只需集合运算 (O(n)) 而非 DP (O(n*m))。
    """
    if length <= 0:
        return bool(a) and bool(b)
    if len(a) < length or len(b) < length:
        return False
    windows = {b[i:i + length] for i in range(len(b) - length + 1)}
    return any(a[i:i + length] in windows for i in range(len(a) - length + 1))


def longest_common_substring_len(a: str, b: str) -> int:
    """最长公共子串长度 (二分 + 窗口集合, 精确值)。

    用于诊断/报错信息 (告诉作者到底抄了几个字), 单测也直接断言它。
    空串返回 0。
    """
    if not a or not b:
        return 0
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _has_common_substring_at_least(a, b, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def max_common_substring_len(query: str, corpus_texts: Iterable[str]) -> int:
    """query 与任一 corpus 文本的最长公共子串 (取最大值); 空语料返回 0。"""
    return max((longest_common_substring_len(query, t or "") for t in corpus_texts), default=0)


def guard_case(
    case_id: str,
    query: str,
    corpus_texts: Sequence[str],
    min_verbatim: int = MIN_COMMON_SUBSTRING,
) -> Dict[str, Any]:
    """单条 case 的守卫结论。

    Returns:
        {"id", "query", "ok", "max_common", "limit"}
        ok=False 表示 query 与语料有 >= min_verbatim 字的公共子串 (抄原文)。
    """
    longest = max_common_substring_len(query, corpus_texts)
    return {
        "id": case_id,
        "query": query,
        "ok": longest < min_verbatim,
        "max_common": longest,
        "limit": min_verbatim,
    }


def guard_cases(
    cases: Sequence[Dict[str, Any]],
    corpus_texts: Sequence[str],
    min_verbatim: int = MIN_COMMON_SUBSTRING,
) -> List[Dict[str, Any]]:
    """批量守卫; 只返回**不合法**的结论列表 (全部合法即空列表)。"""
    results = [
        guard_case(str(c.get("id", "?")), str(c.get("query", "")), corpus_texts, min_verbatim)
        for c in cases
    ]
    return [r for r in results if not r["ok"]]


def format_violations(violations: Sequence[Dict[str, Any]]) -> str:
    """把违规结论格式化成可直接打印的报错文本。"""
    lines = [
        f"反逐字改写守卫未通过: {len(violations)} 条 query 与语料公共子串 >= "
        f"{MIN_COMMON_SUBSTRING} 字 (抄了原文):"
    ]
    for v in violations:
        lines.append(
            f"  - {v['id']}: 最长公共子串 {v['max_common']} 字 / 上限 {v['limit'] - 1} 字 "
            f"| query={v['query']!r}"
        )
    lines.append("  → 请把 query 改写成用户口语提问, 不要照抄语料原文。")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 评估集结构校验 (rag.json / 草稿共用)
# ----------------------------------------------------------------------

def expected_ids_dangling(
    cases: Sequence[Dict[str, Any]], corpus_doc_ids: Iterable[int]
) -> List[Dict[str, Any]]:
    """返回「期望 doc_id 在 corpus 里不存在」的 case 列表 (悬空期望)。"""
    known = {int(x) for x in corpus_doc_ids}
    dangling: List[Dict[str, Any]] = []
    for c in cases:
        missing = [int(x) for x in c.get("expected_doc_ids", []) if int(x) not in known]
        if missing:
            dangling.append({"id": c.get("id"), "missing": missing})
    return dangling


def load_rag_case_file(path: Path) -> Dict[str, Any]:
    """读一份 rag 评估集 JSON (rag.json / rag.corpus.json 同构)。"""
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def corpus_texts_of(data: Dict[str, Any]) -> List[str]:
    """取语料正文列表 (用于守卫)。"""
    return [str(d.get("content", "")) for d in data.get("corpus", [])]


def corpus_doc_ids_of(data: Dict[str, Any]) -> List[int]:
    return [int(d["doc_id"]) for d in data.get("corpus", [])]
