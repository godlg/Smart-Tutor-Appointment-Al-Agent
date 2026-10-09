"""
学科 / 年级抽取器(P0-1)

从会话上下文里用**规则**抽出 subject / grade_level,供 RAG 的
filter-then-vector 硬约束使用。不新增 LLM 调用:纯函数、零成本、可单测。

抽取优先级(先到先得,按字段逐个填充,前面来源已抽到的字段不再被覆盖):
    1. 当前输入 + 本 session 最近 N 条消息(InMemoryChatMessageHistory)
    2. 会话摘要文本(ConversationContext.build 的输出)
    3. 用户画像摘要文本(MemoryContext,含 favorite_subject)
    4. 默认学生画像(StudentProfileService,由调用方注入 provider)

当前输入排在候选文本最前:本轮明确说出的「高一 / 数学」优先于历史里的旧学科,
避免远古历史污染当前查询;同时仍然只抽最近 N 条,不看更早的对话。

本模块属 agents 层:只向下依赖 services 层(默认画像 provider),不 import api / web。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# 学科白名单(子串匹配)。故意不含「数」这类单字,避免「数学」以外的误抽;
# 也不含「学科」「全科」这类跨学科词,跨学科表述不应被当成某个具体学科。
SUBJECT_WHITELIST: Tuple[str, ...] = (
    "数学", "语文", "英语", "物理", "化学", "生物",
    "历史", "地理", "政治", "科学", "信息", "编程",
)

# 年级词 → 年级段。直接命中 小学 / 初中 / 高中 也认(画像里存的就是年级段)。
GRADE_KEYWORD_MAP: Dict[str, str] = {
    "一年级": "小学", "二年级": "小学", "三年级": "小学",
    "四年级": "小学", "五年级": "小学", "六年级": "小学",
    "七年级": "初中", "八年级": "初中", "九年级": "初中",
    "初一": "初中", "初二": "初中", "初三": "初中",
    "高一": "高中", "高二": "高中", "高三": "高中",
    "小学": "小学", "初中": "初中", "高中": "高中",
}

# 长词优先匹配(「一年级」先于「小学」等),避免短词截断长词
_GRADE_KEYWORDS_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(GRADE_KEYWORD_MAP, key=len, reverse=True)
)

# 只抽最近 N 条消息:更早的历史与当前查询无关,抽进来只会污染硬约束
RECENT_MESSAGE_LIMIT = 4

# 占位文案不是真实历史,不能当候选文本(否则「暂无历史消息」会被抽成学科「历史」)
_PLACEHOLDER_TEXTS = (
    "暂无历史消息",
    "[更早的对话已折叠]",
    "[更早的对话已滚动摘要存档]",
)


def extract_subject(text: Optional[str]) -> Optional[str]:
    """从文本里抽学科(白名单子串匹配);无命中返回 None。"""
    if not text:
        return None
    for subject in SUBJECT_WHITELIST:
        if subject in text:
            return subject
    return None


def extract_grade_level(text: Optional[str]) -> Optional[str]:
    """从文本里抽年级段;无命中返回 None。

    「初二」→ 初中、「高一」→ 高中、「六年级」→ 小学,
    直接写「小学 / 初中 / 高中」也认。
    """
    if not text:
        return None
    for keyword in _GRADE_KEYWORDS_BY_LENGTH:
        if keyword in text:
            return GRADE_KEYWORD_MAP[keyword]
    return None


def extract_subject_grade(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """一次性抽 (subject, grade_level);任一字段无命中为 None。"""
    return extract_subject(text), extract_grade_level(text)


def render_recent_messages(messages: Any, limit: int = RECENT_MESSAGE_LIMIT) -> str:
    """把最近 limit 条消息渲染成「用户/顾问: 内容」多行文本。

    messages 为空 / 无 content 时返回空串(调用方据此跳过该来源)。
    """
    if not messages:
        return ""
    items = list(messages)
    if limit and limit > 0:
        items = items[-limit:]
    lines: List[str] = []
    for msg in items:
        content = getattr(msg, "content", None)
        if not content:
            continue
        role = "用户" if getattr(msg, "type", "") == "human" else "顾问"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def resolve_subject_grade(
    query: str,
    *,
    history: Any = None,
    context_texts: Sequence[Optional[str]] = (),
    profile_provider: Optional[
        Callable[[], Tuple[Optional[str], Optional[str]]]
    ] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """按优先级抽 (subject, grade_level);全无命中返回 (None, None)。

    Args:
        query: 当前用户输入(最高优先,排在同源文本最前)
        history: 本 session 的 InMemoryChatMessageHistory(取最近 N 条)
        context_texts: 其它候选文本(会话摘要 / 画像摘要),按优先级排列
        profile_provider: 最后兜底(默认学生画像);None 表示不兜底
    """
    candidates: List[str] = []
    if query:
        candidates.append(query)
    recent = render_recent_messages(getattr(history, "messages", None))
    if recent:
        candidates.append(recent)
    for text in context_texts:
        if text:
            candidates.append(text)

    subject: Optional[str] = None
    grade_level: Optional[str] = None
    for text in candidates:
        if _is_placeholder(text):
            continue
        if subject is None:
            subject = extract_subject(text)
        if grade_level is None:
            grade_level = extract_grade_level(text)
        if subject is not None and grade_level is not None:
            break

    if subject is None and grade_level is None and profile_provider is not None:
        try:
            return profile_provider()
        except Exception:
            # 兜底画像不可用(无表 / DB 异常)不应拖垮检索:退回全量
            return None, None

    return subject, grade_level


def default_profile_filters() -> Tuple[Optional[str], Optional[str]]:
    """优先级 4 的兜底实现:默认学生画像的学科 / 年级段。

    只读(默认家长 / 默认学生由 app 启动时创建);任何异常都返回 (None, None),
    保证检索链路不会因为画像不可用而失败。
    """
    try:
        from services.student_profile_service import StudentProfileService

        service = StudentProfileService()
        parent = service.get_or_create_default_parent()
        students = service.get_students_by_parent(parent["id"])
        if not students:
            return None, None

        student = students[0]
        weak_subjects = student.get("weak_subjects") or []
        raw_subject = weak_subjects[0] if weak_subjects else None
        raw_grade = student.get("grade_level")
        # 画像字段可能带具体年级词(如「高一」),统一按规则归一成年级段
        return (
            extract_subject(raw_subject) or raw_subject,
            extract_grade_level(raw_grade) or raw_grade,
        )
    except Exception:
        return None, None


def _is_placeholder(text: str) -> bool:
    """占位文案(无历史 / 折叠 / 滚动摘要)不算候选文本。"""
    stripped = text.strip()
    return any(stripped == p for p in _PLACEHOLDER_TEXTS)
