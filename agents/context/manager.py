"""
上下文管理器

统一按优先级组装各上下文,产出 LLM 的完整大输入文本。
提供近似 token 估算与各上下文用量报告,便于后续做上下文工程观测与告警。
"""

import logging
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Dict, Any, Callable

from .base import BaseContext
from .types import ContextBudgetExceeded, estimate_token_count
from .contexts.conversation import ConversationContext, _FOLD_PLACEHOLDER
from .contexts.memory import MemoryContext

logger = logging.getLogger(__name__)

# Optional budgets: recent dialogue receives the largest share. Unused shares are
# redistributed; required instructions and the current question never participate.
_OPTIONAL_SHARES = {"conversation": 0.45, "summary": 0.20,
                    "knowledge": 0.20, "memory": 0.15}
_SUMMARY_HEADER = "【近期滚动摘要】"
_SUMMARY_PREFIX = "早期对话摘要: [extractive 摘录] "
_SUMMARY_KEYWORDS = ("确认", "只能", "不能", "必须", "不要", "约束", "时间", "周",
                     "老师", "年级", "数学", "英语", "语文", "物理", "化学",
                     "学科", "待办", "预约", "试讲", "线上", "线下")


@dataclass(frozen=True)
class _PromptBlock:
    name: str
    text: str
    priority: int
    category: str = "required"


class ContextManager:
    """上下文管理器

    持有按 priority 升序排列的上下文集合,每次 build 产出完整输入文本。
    build 时把 session_id / task 暴露给各上下文(如会话历史、用户记忆按会话聚合),
    也支持传入额外临时上下文(如 RAG 检索结果)。

    分层记忆(60% 窗口滚动摘要):
    - rollup_enabled=True 时,build 会在「窗口 token 占用 ≥ rollup_window_ratio」时
      触发滚动摘要:把会话历史最旧若干轮折叠成一句摘要,写回 long_term_memories(summary),
      并替换上下文中的原文(占位)。
    - 默认关闭,不影响原有 max_tokens 压缩兜底行为。
    """

    def __init__(
        self,
        contexts: List[BaseContext] = None,
        rollup_enabled: bool = False,
        rollup_window_ratio: float = 0.6,
        rollup_rounds: int = 4,
        rollup_persister: Optional[Callable] = None,
        identity=None,
        conversation_service=None,
        token_counter: Optional[Callable[[str], int]] = None,
        output_tokens: int = 0,
        summarizer: Optional[Callable] = None,
        summary_max_chars: int = 600,
        keep_recent_rounds: int = 4,
        memory_service=None,
    ):
        self.contexts = sorted(contexts or [], key=lambda c: (c.priority, c.name))
        self.identity = identity
        self.conversation_service = conversation_service
        self.memory_service = memory_service
        self.session_id: Optional[str] = getattr(identity, "session_id", None)
        self.task: str = ""
        self.max_tokens: Optional[int] = None  # Legacy callers remain unbounded.
        self.token_counter = token_counter or estimate_token_count
        self.output_tokens = max(0, output_tokens)
        self.summarizer = summarizer
        self.summary_max_chars = max(1, summary_max_chars)
        self.keep_recent_rounds = max(1, keep_recent_rounds)
        self._last_build = ""
        self._last_blocks: List[_PromptBlock] = []
        self._history_snapshots: Dict[str, list] = {}
        self._summary_records: Dict[str, list] = {}
        # 滚动摘要配置(60% 窗口,可配)
        self.rollup_enabled = rollup_enabled
        self.rollup_window_ratio = rollup_window_ratio
        self.rollup_rounds = rollup_rounds
        # 摘要持久化器:默认 LongTermMemoryService().save_rollup_summary,可注入(mock/测试)
        self._rollup_persister = rollup_persister
        # 最近一次 build 是否触发了滚动摘要(供测试/观测断言)
        self.last_rollup_triggered = False

    def add(self, context: BaseContext) -> None:
        """追加一个上下文并立即按 priority 重排。"""
        self.contexts = sorted([*self.contexts, context], key=lambda c: (c.priority, c.name))

    def build(
        self,
        task: str = "",
        session_id: Optional[str] = None,
        extra_contexts: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """组装所有上下文(按 priority)→ 完整大输入文本。

        Args:
            task: 用户当前输入,拼在文本末尾作为待回答问题
            session_id: 会话 id(供会话历史等按会话聚合的上下文使用)
            extra_contexts: 额外临时上下文({name: text}),如 RAG 检索结果,
                            按给定顺序追加在 task 之前。
            max_tokens: Token 阈值。组装后若 estimate_tokens 超过该值,
                        触发 _compress 对会话历史做裁剪再重组。

        Returns:
            完整注入文本
        """
        self.session_id = session_id or self.session_id
        self.task = task or ""
        self.last_rollup_triggered = False
        self.validate_identity_contexts()
        self._restore_summaries()
        blocks = self._collect_blocks(extra_contexts)
        limit = self.max_tokens if max_tokens is None else max_tokens
        budget = None if limit is None else max(0, limit - self.output_tokens)
        # Reject impossible mandatory payloads before making persistence changes.
        if budget is not None:
            self._check_required(blocks, budget)
        if (self.rollup_enabled and budget is not None and budget > 0
                and self.token_counter(self._join(blocks)) >= budget * self.rollup_window_ratio):
            blocks = self._rollup_blocks(blocks)
        if budget is not None:
            blocks = self._fit_budget(blocks, budget)
        result = self._join(blocks)
        if budget is not None and self.token_counter(result) > budget:
            raise ContextBudgetExceeded("必需上下文无法容纳，请缩短输入或提高预算。",
                                        self.token_counter(result), budget)
        self._last_blocks = blocks
        self._last_build = result
        return result

    @staticmethod
    def _join(blocks: List[_PromptBlock]) -> str:
        return "\n\n".join(block.text for block in blocks if block.text)

    @staticmethod
    def _category(name: str) -> str:
        # Unknown extra blocks are business instructions, not assumed RAG.
        key = name.lower()
        if key in ("conversation", "history"):
            return "conversation"
        if key in ("memory", "knowledge", "summary"):
            return key
        return "required"

    def _collect_blocks(self, extra_contexts: Optional[Dict[str, str]]) -> List[_PromptBlock]:
        blocks = []
        self._history_snapshots = {}
        for ctx in self.contexts:
            category = self._category(ctx.name)
            try:
                text = ctx.build(self) or ""
                if isinstance(ctx, ConversationContext) and self.session_id:
                    self._history_snapshots[ctx.name] = list(ctx.get_history(self.session_id).messages)
            except Exception:
                # Known system/tool/schema instructions cannot silently disappear.
                if ctx.name in ("system_prompt", "sys", "tools", "structured_output"):
                    raise
                logger.warning("上下文 %s 构建失败，跳过可选内容", ctx.name, exc_info=True)
                text = ""
            blocks.append(_PromptBlock(ctx.name, text, ctx.priority, category))
        for name, text in (extra_contexts or {}).items():
            if text:
                blocks.append(_PromptBlock(name, text, 50, self._category(name)))
        summary = self._summary_block()
        if summary:
            blocks.append(summary)
        if self.task:
            blocks.append(_PromptBlock("task", f"用户问题：{self.task}\n\n请回答用户的问题。", 100))
        return blocks

    def validate_identity_contexts(self) -> None:
        if self.identity is None:
            return
        expected = (self.identity.user_id, self.identity.student_id)
        for context in self.contexts:
            if (isinstance(context, ConversationContext) and context.history_provider is not None
                    and self.conversation_service is None):
                raise ValueError("身份上下文的共享历史要求可校验的会话存储")
            if not isinstance(context, MemoryContext):
                continue
            bound = context.identity
            if bound is None or (bound.user_id, bound.student_id) != expected:
                raise ValueError("记忆 context scope 与 manager 身份不一致")

    def _restore_summaries(self) -> None:
        if self.conversation_service is None or self.identity is None:
            return
        if self.session_id != self.identity.session_id:
            raise ValueError("ContextManager session 与已验证 identity 不一致")
        # Permission must be confirmed before optional reads/cache access. A DB
        # outage during authorization is indeterminate, never a safe fallback.
        self.conversation_service.ensure_session(self.identity)
        from services.identity_service import SessionOwnershipError, StudentClaimError
        try:
            summaries = self.conversation_service.get_summaries(self.identity)
            watermark = self.conversation_service.summary_watermark(self.identity)
        except (SessionOwnershipError, StudentClaimError):
            # Auth/binding failure is not an optional-summary outage. Reading the
            # cached history afterwards would expose the owner's private text.
            raise
        except Exception:
            logger.warning("恢复滚动摘要失败，保留原始历史", exc_info=True)
            return
        self._summary_records = {**self._summary_records, self.session_id: summaries}
        for conv in self.contexts:
            if isinstance(conv, ConversationContext):
                conv.mark_rolled_up(self.session_id, watermark)

    def build_retrieval_context(self, max_chars: int = 1600) -> str:
        """有界 RAG 历史视图:近期摘要独立配额 + 最近轮次,不做记忆 embedding。

        检索发生在最终 prompt build 前,因此必须自己恢复摘要水位并限制长消息。
        不触发 rollup 写入,不包含系统/工具指令,不无限拼接全历史。
        """
        if max_chars <= 0:
            return ""
        self.validate_identity_contexts()
        self._restore_summaries()
        summary = self._summary_block()
        summary_text = summary.text[:min(self.summary_max_chars, max_chars // 2)] if summary else ""
        conv = next((ctx for ctx in self.contexts if isinstance(ctx, ConversationContext)), None)
        history_text = ""
        if conv is not None and self.session_id:
            from .contexts.conversation import _NO_HISTORY
            messages = conv.get_history(self.session_id).messages
            history_text = conv.render_history(messages, conv._rolled_up_count(self.session_id),
                                               self.keep_recent_rounds)
            if history_text == _NO_HISTORY:
                history_text = ""
        separator = "\n\n" if summary_text and history_text else ""
        remaining = max(0, max_chars - len(summary_text) - len(separator))
        recent = history_text[-remaining:] if remaining else ""
        return summary_text + separator + recent

    def _summary_block(self) -> Optional[_PromptBlock]:
        records = self._summary_records.get(self.session_id, [])
        if not records:
            return None
        # A bounded recent-text view, never a concatenating recursive summary.
        parts, remaining = [], self.summary_max_chars
        for record in reversed(records):
            text = record["content"]
            if remaining <= 0:
                break
            parts.append(text[:remaining])
            remaining -= len(parts[-1]) + 1
        # Latest-first also makes prefix budget clipping retain corrections.
        body = "\n".join(parts)
        return _PromptBlock("rollup_summary", f"{_SUMMARY_HEADER}\n{body}", 25, "summary")

    def _summarize(self, messages: list) -> str:
        if self.summarizer is not None:
            summary = self.summarizer(messages, self.summary_max_chars)
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("summarizer 必须返回非空文本")
            return summary[:self.summary_max_chars]
        candidates = []
        for msg in messages:
            role = "用户" if msg.type == "human" else "顾问"
            for sentence in re.split(r"[。！？；，,\n]+", str(msg.content)):
                if not sentence.strip():
                    continue
                score = sum(word in sentence for word in _SUMMARY_KEYWORDS)
                score += 2 * bool(re.search(r"\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}", sentence))
                candidates.append((score, len(candidates), f"{role}: {sentence.strip()}"))
        remaining = max(0, self.summary_max_chars - len(_SUMMARY_PREFIX))
        selected = []
        # Bound any single excerpt so a long confirmed utterance does not consume
        # all space before grade/subject/teacher/time from the same segment.
        excerpt_limit = max(24, remaining // max(1, min(len(candidates), 6)))
        for _score, index, text in sorted(candidates, key=lambda row: (-row[0], row[1])):
            if remaining <= 0:
                break
            clipped = text[:min(remaining, excerpt_limit)]
            selected.append((index, clipped))
            remaining -= len(clipped) + 1
        body = "；".join(text for _, text in sorted(selected))
        return (_SUMMARY_PREFIX + body)[:self.summary_max_chars]

    def _persist_memory(self, summary: str, conv: ConversationContext) -> bool:
        user_id = getattr(self.identity, "user_id", None) or getattr(conv, "user_id", None) or "default_user"
        if self._rollup_persister is not None:
            # Do not add kwargs to legacy callbacks or retry on TypeError.
            return self._rollup_persister(user_id=user_id, summary_text=summary,
                                          session_id=self.session_id) is not False
        service = self.memory_service
        owns_service = service is None
        if service is None:
            if self.identity is not None:
                raise ValueError("身份摘要要求注入 conversation_service/memory_service 或 persister")
            from services.long_term_memory_service import LongTermMemoryService
            service = LongTermMemoryService()  # Legacy no-identity path only.
        try:
            kwargs = {"user_id": user_id, "memory_type": "summary", "content": summary,
                      "session_id": self.session_id, "auto_embed": False}
            if self.identity is not None:
                kwargs["student_id"] = self.identity.student_id
            return service.add_memory(**kwargs) is not False
        finally:
            if owns_service:
                service.db_router.close()

    def _save_rollup(self, summary: str, start: int, end: int, conv: ConversationContext) -> bool:
        if self.conversation_service is None or self.identity is None:
            return self._persist_memory(summary, conv)
        # ConversationService owns atomic summary + long-term memory + watermark
        # archival on its shared router. Never copy into another default database.
        return self.conversation_service.save_rollup(self.identity, start, end, summary)

    def _rollup_blocks(self, blocks: List[_PromptBlock]) -> List[_PromptBlock]:
        conv = next((ctx for ctx in self.contexts if isinstance(ctx, ConversationContext)), None)
        if conv is None or not self.session_id:
            return blocks
        messages = self._history_snapshots.get(conv.name, [])
        rolled = conv._rolled_up_count(self.session_id)
        segment_size = max(2, self.rollup_rounds * 2)
        eligible = max(0, len(messages) - self.keep_recent_rounds * 2)
        if eligible - rolled < segment_size:
            return blocks
        end = rolled + segment_size
        try:
            summary = self._summarize(messages[rolled:end])
            if not self._save_rollup(summary, rolled + 1, end, conv):
                raise RuntimeError("rollup persister returned False")
        except Exception:
            logger.warning("滚动摘要持久化失败，水位不推进，保留未存档原文", exc_info=True)
            # A subsequent hard-budget pass must not silently hide this failed segment.
            return [replace(block, category="required") if block.name == conv.name else block
                    for block in blocks]
        conv.mark_rolled_up(self.session_id, end)
        self.last_rollup_triggered = True
        records = [*self._summary_records.get(self.session_id, []),
                   {"start_seq": rolled + 1, "end_seq": end, "content": summary}]
        self._summary_records = {**self._summary_records, self.session_id: records}
        replaced = [replace(block, text=conv.render_history(messages, end))
                    if block.name == conv.name else block
                    for block in blocks if block.name != "rollup_summary"]
        summary_block = self._summary_block()
        # Summary before current question, independent from memory Top-k.
        index = next((i for i, block in enumerate(replaced) if block.name == "task"), len(replaced))
        return [*replaced[:index], summary_block, *replaced[index:]]

    def _check_required(self, blocks: List[_PromptBlock], budget: int) -> None:
        required = [block for block in blocks if block.category == "required"]
        used = self.token_counter(self._join(required))
        if used > budget:
            raise ContextBudgetExceeded("必需上下文超出预算，请缩短输入或提高预算。", used, budget)

    def _clip_text(self, text: str, allowance: int, category: str) -> str:
        if self.token_counter(text) <= allowance:
            return text
        if allowance <= 0:
            return ""
        if category == "conversation":
            headers = [line for line in text.splitlines() if line.startswith("[更早的对话")]
            marker = "\n".join([*headers, _FOLD_PLACEHOLDER]) + "\n"
            body = "\n".join(line for line in text.splitlines() if not line.startswith("[更早的对话"))
            make_candidate = lambda count: marker + (body[-count:] if count else "")
        elif category == "summary":
            marker = _SUMMARY_HEADER + "\n"
            body = text.removeprefix(marker)
            make_candidate = lambda count: marker + body[:count] + "…"
        else:
            marker = ""
            body = text
            make_candidate = lambda count: body[:count] + "…"
        if self.token_counter(make_candidate(0)) > allowance:
            return ""
        lo, hi = 0, len(body)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter(make_candidate(mid)) <= allowance:
                lo = mid
            else:
                hi = mid - 1
        candidate = make_candidate(lo)
        return candidate if self.token_counter(candidate) <= allowance else ""

    def _fit_budget(self, blocks: List[_PromptBlock], budget: int) -> List[_PromptBlock]:
        self._check_required(blocks, budget)
        if self.token_counter(self._join(blocks)) <= budget:
            return blocks
        # First remove whole older rounds using the history snapshot, no ctx.build.
        compact = []
        for block in blocks:
            if block.category == "conversation" and block.name in self._history_snapshots:
                conv = next(ctx for ctx in self.contexts if ctx.name == block.name)
                text = conv.render_history(self._history_snapshots[block.name],
                                           conv._rolled_up_count(self.session_id),
                                           self.keep_recent_rounds)
                compact.append(replace(block, text=text))
            else:
                compact.append(block)
        if self.token_counter(self._join(compact)) <= budget:
            return compact
        required = [block for block in compact if block.category == "required"]
        # Include joining overhead; final full recount handles non-additive counters.
        available = max(0, budget - self.token_counter(self._join(required))
                        - self.token_counter("\n\n") * len(compact))
        optional = [block for block in compact if block.category != "required" and block.text]
        weights = sum(_OPTIONAL_SHARES[block.category] for block in optional)
        fitted = [replace(block, text="") if block.category != "required" else block for block in compact]
        # Initial independent category shares; then distribute any unused allowance.
        for i, block in enumerate(compact):
            if block.category != "required" and block.text:
                share = int(available * _OPTIONAL_SHARES[block.category] / weights) if weights else 0
                fitted[i] = replace(block, text=self._clip_text(block.text, share, block.category))
        for i in sorted(range(len(compact)), key=lambda i: (
                {"conversation": 0, "summary": 1, "knowledge": 2, "memory": 3}.get(compact[i].category, 4))):
            block = compact[i]
            if block.category == "required":
                continue
            slack = max(0, budget - self.token_counter(self._join(fitted)))
            allowance = self.token_counter(fitted[i].text) + slack
            fitted[i] = replace(block, text=self._clip_text(block.text, allowance, block.category))
        # Enforce final joined count even when the injected counter is non-additive.
        for category in ("knowledge", "memory", "summary", "conversation"):
            for i, block in enumerate(fitted):
                if self.token_counter(self._join(fitted)) <= budget:
                    return fitted
                if block.category == category:
                    fitted[i] = replace(block, text="")
        self._check_required(fitted, budget)
        return fitted

    def estimate_tokens(self) -> int:
        """按注入计数器统计最近一次成功 build 的完整文本。"""
        return self.token_counter(self._last_build)

    def usage_report(self) -> dict:
        """各上下文的用量报告。

        Returns:
            {name: {"chars": n, "priority": p, "preview": 前50字}}

        用于观测每个上下文对输入体积的贡献,定位"上下文膨胀"点。
        """
        report: Dict[str, Any] = {}
        cached = {block.name: block for block in self._last_blocks}
        for ctx in self.contexts:
            block = cached.get(ctx.name)
            text = block.text if block else ""
            report[ctx.name] = {
                "chars": len(text),
                "priority": ctx.priority,
                "preview": text[:50],
            }
        return report