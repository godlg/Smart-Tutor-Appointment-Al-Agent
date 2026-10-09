"""
用户记忆上下文（分层记忆增强版）

两层记忆注入:
1. 长期记忆召回: 有 task(用户 query)时,按「0.6*向量相似度 + 0.3*时间衰减 + 0.1*重要度」
   从 long_term_memories 召回 Top-5 相关记忆,渲染成「相关记忆」注入(记忆摘要);
2. 画像摘要(原有行为,priority=20): 复用 pattern_analyzer 的用户画像,
   作为兜底与召回记忆共存。

兼容性(不破坏原行为):
- 无 task / 无记忆 / embedding 缺失 / service 不可用时 → 回退原「用户画像摘要」
- 召回失败只记录日志,绝不让上游崩溃
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
_VECTOR_QUOTA = 5
_SUMMARY_QUOTA = 3

from ..base import BaseContext
from ..types import PRIORITY_MEMORY, DEFAULT_USER_ID

# 无画像时的占位文案
_NO_PROFILE = "暂无用户画像(该用户尚未积累足够的上课记录)"


def _days_since(value) -> Optional[int]:
    """把 last_lesson_date(字符串或 datetime)换算成距今天数。

    解析失败或为空时返回 None,容忍各种存储格式,不让异常穿透。
    """
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        if isinstance(value, str):
            # 兼容 "2026-09-01" / "2026-09-01T10:30:00" / 带 Z 的 UTC 等
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = value

        # 统一转成 naive 后再与现时刻比较,避免 aware/naive 混算是抛错
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            dt = dt.replace(tzinfo=None)
        delta = datetime.now(timezone.utc).replace(tzinfo=None) - dt
        return max(0, delta.days)
    except Exception:
        return None


class MemoryContext(BaseContext):
    """用户记忆上下文:长期记忆向量召回 + 用户画像摘要(priority=20)。"""

    name = "memory"
    priority = PRIORITY_MEMORY

    def __init__(self, analyzer=None, user_id: str = DEFAULT_USER_ID,
                 memory_service=None, identity=None):
        """分析器 / 记忆服务均可注入(便于测试 mock),默认懒加载真实实现。

        注意:PatternAnalyzer 必须携带 behavior_service,否则 analyze_user_preferences
        内部走 behavior_db 分支时 behavior_db 为 None 会抛 AttributeError
        (曾经的真实 bug:画像永远拿不到,只报"分析家长偏好失败")。
        memory_service 缺省时懒加载 LongTermMemoryService,仅在有 task 时才触碰。
        """
        self.analyzer = analyzer
        self.identity = identity
        self.user_id = identity.user_id if identity is not None else user_id
        self._memory_service = memory_service

    # 懒加载:不注入时延后创建,避免构造阶段就打开数据库(测试友好)
    def _get_analyzer(self):
        if self.analyzer is None:
            from agents.user_behavior.pattern_analyzer import PatternAnalyzer
            if self._memory_service is not None:
                router = getattr(self._memory_service, "db_router", None)
                if router is None:
                    raise ValueError("注入记忆服务未提供共享行为 router，跳过画像")
                self.analyzer = PatternAnalyzer(router.user_behavior_repo)
            elif self.identity is not None:
                raise ValueError("已验证身份缺少共享存储，跳过画像")
            else:
                # Only the explicitly legacy, no-identity path may use defaults.
                from services.user_behavior_service import UserBehaviorService
                self.analyzer = PatternAnalyzer(UserBehaviorService())
        return self.analyzer

    def _get_memory_service(self):
        if self._memory_service is None:
            if self.identity is not None:
                raise ValueError("已验证身份缺少共享记忆服务，跳过召回")
            from services.long_term_memory_service import LongTermMemoryService
            self._memory_service = LongTermMemoryService()
        return self._memory_service

    # ------------------------------------------------------------------
    # build 入口
    # ------------------------------------------------------------------

    def build(self, manager) -> str:
        """两段式注入:相关记忆(有 query 时) + 用户画像摘要(兜底/共存)。"""
        blocks: List[str] = []

        recalled = self._recall_memories(manager)
        if recalled:
            blocks.append(recalled)

        profile = self._build_profile()
        if profile:
            blocks.append(profile)

        return "\n\n".join(blocks) if blocks else _NO_PROFILE

    # ------------------------------------------------------------------
    # 向量召回注入
    # ------------------------------------------------------------------

    def _recall_memories(self, manager) -> str:
        """按 manager 暴露的 task(若有)召回 Top-5 相关记忆,渲染成摘要。

        无 query 时不做向量召回;已验证 scope 的文本摘要仍有独立配额。
        向量错误不阻断摘要读取;无可用记忆时交由画像兜底。
        """
        task = getattr(manager, "task", "") or ""
        has_query = bool(task.strip())
        if not has_query and self.identity is None:
            return ""

        try:
            service = self._get_memory_service()
        except Exception:
            logger.warning("记忆依赖不可用，跳过召回", exc_info=True)
            return ""
        scope = {"student_id": self.identity.student_id} if self.identity else {}
        try:
            memories = (service.retrieve_top_k(query=task, user_id=self.user_id, **scope)[:_VECTOR_QUOTA]
                        if has_query else [])
        except Exception:
            logger.warning("向量召回失败，继续尝试文本摘要", exc_info=True)
            memories = []
        summaries = []
        if self.identity is not None:
            try:
                # Independent quota: a full vector Top-5 cannot displace text archives.
                summaries = service.get_user_memories(
                    self.user_id, "summary", limit=_SUMMARY_QUOTA + _VECTOR_QUOTA, **scope)
            except Exception:
                logger.warning("文本摘要读取失败，保留可用向量召回", exc_info=True)
        recalled_ids = {m.get("id") for m in memories if m.get("id") is not None}
        summaries = [m for m in summaries
                     if m.get("id") is None or m.get("id") not in recalled_ids]
        lines = []
        # Prefix clipping under the optional memory budget should favor the
        # deterministic text archive over arbitrarily long vector-hit bodies.
        if summaries:
            lines.append("【近期记忆摘要】")
            lines.extend(f"- {memory.get('content', '')}" for memory in summaries[:_SUMMARY_QUOTA])
        if memories:
            lines.append("【相关记忆】")
            for memory in memories:
                lines.append(f"- ({memory.get('score', 0.0):.2f}) {memory.get('content', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 画像摘要(原有行为)
    # ------------------------------------------------------------------

    def build_profile_summary(self) -> str:
        """只渲染用户画像摘要(不做长期记忆向量召回),供轻量场景复用。

        RAG 的学科抽取(P0-1)只需要画像里的学科偏好,不应为它多付一次
        向量召回/嵌入调用;异常与 _build_profile 一致地回退空串。
        """
        return self._build_profile()

    def _build_profile(self) -> str:
        """把用户画像渲染成摘要;无数据 / 异常时返回空串。"""
        try:
            scope = self.identity.memory_scope() if self.identity else self.user_id
            prefs = self._get_analyzer().analyze_user_preferences(scope)
        except Exception:
            logger.warning("画像读取失败，跳过可选画像", exc_info=True)
            prefs = None

        if not prefs:
            return ""

        lines = ["【用户画像】"]
        lines.append(f"偏好老师ID: {prefs.get('favorite_teacher_id') or '未知'}")
        lines.append(f"偏好学科: {prefs.get('favorite_subject') or '未知'}")
        lines.append(
            f"常用时长: {prefs.get('favorite_duration') if prefs.get('favorite_duration') is not None else '未知'} 分钟"
        )
        lines.append(f"累计正式课: {prefs.get('total_lessons') or 0} 节")
        days = _days_since(prefs.get("last_lesson_date"))
        lines.append(f"距上次上课: {days if days is not None else '未知'} 天")
        return "\n".join(lines)
