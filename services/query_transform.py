# services/query_transform.py
"""检索前的 Query 变换(三档 off / rule / llm)。

用户口语化、指代不清的 query 直接送检索时,Dense 与 BM25 都容易漏召回。
本模块在检索前把一条 query 变换成 1~max_variants 条检索串,由上层多路召回后
按 RRF 融合(见 HybridRetriever.search_multi / KnowledgeService.search_multi)。

硬契约(任何 mode、任何异常下都成立):
    1. 返回值非空列表,且 [0] == 原始 query(原串永远排第 0 位且必须被检索);
    2. 无重复元素;
    3. len(result) <= settings.query_transform_max_variants(含原串)。

三档:
- off  : 直接 [query],零变换。
- rule : 零 LLM 成本的两条规则 —— 术语归一(口语词 → 规范词)+ 指代补全
         (指代词命中且上下文能抽出「年级 + 学科」时补成「年级学科：query」)。
- llm  : 额外调一次小 LLM 产最多 2 条改写,与 rule 变体合并去重;
         超时 / 异常 / 解析失败一律回退 rule 结果(不抛异常)。

规则部分与 agents.consultant.subject_extractor 同样风格:纯函数、零 LLM、可单测。
本模块属 services 层:只依赖 config / 标准库 / langchain 现有件,不 import agents。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 规则一: 术语归一(口语词 → 规范词)
# ----------------------------------------------------------------------

# 顺序即替换顺序:先到先替换,替换结果继续参与后续匹配。
TERM_MAP: Tuple[Tuple[str, str], ...] = (
    ("体验课", "试讲"),
    ("试听", "试讲"),
    ("试课", "试讲"),
    ("多少钱", "收费标准 价格"),
    ("贵不贵", "收费标准 价格"),
    ("费用高", "收费标准 价格"),
    ("价钱", "收费标准 价格"),
    ("1对1", "一对一"),
    ("一对1", "一对一"),
    ("提高分数", "提高成绩"),
    ("涨分", "提高成绩"),
    ("提分", "提高成绩"),
    ("辅导班", "辅导"),
    ("补习", "辅导"),
    ("补课", "辅导"),
    ("线上课", "授课形式"),
    ("线下课", "授课形式"),
    ("网课", "授课形式"),
    ("跟不上", "学习困难"),
    ("学不会", "学习困难"),
    ("不行", "学习困难"),
)

# 规则二: 指代补全的触发词。命中其一才考虑补「年级学科」前缀。
PRONOUN_HINTS: Tuple[str, ...] = ("这个", "那个", "它", "这门课", "怎么样", "咋样")

# LLM 变体长度上限(超过即丢弃,避免把检索串写成长篇解释)
LLM_VARIANT_MAX_CHARS = 60
# LLM 请求的变体条数(不含原串)
LLM_MAX_VARIANTS = 2

_VALID_MODES = ("off", "rule", "llm")
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _resolve_mode(mode: Optional[str]) -> str:
    """mode 为空时读 settings;非法值回退 rule(不抛异常)。"""
    resolved = (mode if mode is not None else settings.query_transform_mode) or "rule"
    resolved = str(resolved).strip().lower()
    if resolved not in _VALID_MODES:
        logger.warning("QUERY_TRANSFORM=%r 非法, 回退 rule", resolved)
        return "rule"
    return resolved


def _max_variants() -> int:
    """变体数上限(含原串),最小 1。"""
    try:
        return max(1, int(settings.query_transform_max_variants))
    except (TypeError, ValueError):
        return 3


def _timeout() -> float:
    try:
        return max(0.1, float(settings.query_transform_timeout))
    except (TypeError, ValueError):
        return 3.0


def _assemble(query: str, candidates: List[str], max_variants: int) -> List[str]:
    """原串恒在 [0] + 去重 + 截断到 max_variants;保证非空(max_variants<=1 时只有原串)。"""
    out: List[str] = [query]
    for item in candidates:
        if len(out) >= max_variants:
            break
        if not item or item in out:
            continue
        out.append(item)
    return out


# ----------------------------------------------------------------------
# 规则档
# ----------------------------------------------------------------------

def _term_variant(query: str) -> Optional[str]:
    """术语归一:命中口语词 → 替换成规范词的变体;无命中返回 None。"""
    out = query
    hit = False
    for spoken, canonical in TERM_MAP:
        if spoken in out:
            out = out.replace(spoken, canonical)
            hit = True
    if not hit or out == query:
        return None
    return out


def _context_subject_grade(
    context_text: str, history_text: str
) -> Tuple[Optional[str], Optional[str]]:
    """从上下文文本里抽 (subject, grade_level);任一抽不到就是 None。

    两个字段都抽到才允许产变体 —— 抽不到就不产,不许编。
    """
    from agents.consultant.subject_extractor import (
        extract_grade_level,
        extract_subject,
    )

    subject: Optional[str] = None
    grade_level: Optional[str] = None
    for text in (context_text, history_text):
        if not text:
            continue
        if subject is None:
            subject = extract_subject(text)
        if grade_level is None:
            grade_level = extract_grade_level(text)
        if subject is not None and grade_level is not None:
            break
    return subject, grade_level


def _pronoun_variant(
    query: str, context_text: str, history_text: str
) -> Optional[str]:
    """指代补全:命中指代词且上下文能抽出「年级 + 学科」→ 补前缀变体。"""
    if not any(hint in query for hint in PRONOUN_HINTS):
        return None
    if not (context_text or history_text):
        return None
    subject, grade_level = _context_subject_grade(context_text, history_text)
    if subject is None or grade_level is None:
        return None
    variant = f"{grade_level}{subject}：{query}"
    return variant if variant != query else None


def variants(
    query: str,
    *,
    context_text: str = "",
    history_text: str = "",
    mode: Optional[str] = None,
) -> List[str]:
    """同步档(off / rule)。

    返回**必含原始 query 且 index 0 == query**,去重,长度 <= max_variants。
    mode=llm 传进同步档时按 rule 处理(同步档不调 LLM)。
    """
    max_variants = _max_variants()
    if _resolve_mode(mode) == "off":
        return _assemble(query, [], max_variants)

    candidates: List[str] = []
    term = _term_variant(query)
    if term:
        candidates.append(term)
    pronoun = _pronoun_variant(query, context_text, history_text)
    if pronoun:
        candidates.append(pronoun)
    return _assemble(query, candidates, max_variants)


# ----------------------------------------------------------------------
# LLM 档
# ----------------------------------------------------------------------

_LLM_PROMPT = (
    "你是检索式改写助手。把用户的口语化问题改写成最多 {n} 条更适合知识库检索的"
    "检索式(保留关键术语,补全指代)。\n"
    "只输出 JSON 数组,每项一条检索式字符串,不要解释、不要 markdown 代码块。\n"
    "用户问题: {query}"
)


def _response_text(response: Any) -> str:
    """把 LLM 响应取成纯文本(langchain AIMessage / 纯字符串桩都兼容)。"""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # 部分 provider 返回 [{type: text, text: ...}] 结构
        return " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _parse_llm_variants(raw: str, query: str) -> List[str]:
    """从 LLM 文本里解析检索式列表;解析不出 / 为空 → 返回 []。

    丢弃:非字符串项、空串、长度 > LLM_VARIANT_MAX_CHARS、等于原 query 的项。
    """
    text = (raw or "").strip()
    if not text:
        return []
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, list):
        return []

    out: List[str] = []
    for item in payload:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate or candidate == query:
            continue
        if len(candidate) > LLM_VARIANT_MAX_CHARS:
            continue
        out.append(candidate)
    return out


async def _llm_variants(
    query: str, llm: Any, timeout: float
) -> List[str]:
    """调一次 LLM 产变体;超时 / 异常由调用方兜(这里不吞也不重试)。"""
    prompt = _LLM_PROMPT.format(n=LLM_MAX_VARIANTS, query=query)
    response = await asyncio.wait_for(
        llm.ainvoke(prompt), timeout=timeout,
    )
    return _parse_llm_variants(_response_text(response), query)


async def variants_async(
    query: str,
    *,
    context_text: str = "",
    history_text: str = "",
    mode: Optional[str] = None,
    llm: Any = None,
    timeout: Optional[float] = None,
) -> List[str]:
    """异步档。

    mode=llm 时调 LLM 生成变体;超时 / 异常 / 解析失败**一律回退**到 rule 档结果
    (rule 也为空则 [query])。永不抛异常。
    """
    max_variants = _max_variants()
    resolved = _resolve_mode(mode)
    rule_result = variants(
        query, context_text=context_text, history_text=history_text,
        mode="rule" if resolved == "llm" else resolved,
    )
    if resolved != "llm":
        return rule_result

    try:
        if llm is None:
            from config.model_provider import create_chat_model

            llm = create_chat_model(temperature=0)
        extra = await _llm_variants(query, llm, timeout or _timeout())
    except asyncio.TimeoutError:
        logger.warning("query 变换 LLM 超时, 回退 rule 变体")
        return rule_result
    except Exception as exc:  # noqa: BLE001 - 变换失败绝不影响检索
        logger.warning("query 变换 LLM 失败, 回退 rule 变体: %s", exc)
        return rule_result

    # 原串永远第 0 位:LLM 变体排在 rule 变体之后
    return _assemble(query, rule_result[1:] + extra, max_variants)
