"""
轻量 Trace span 工具 (补4: Traces)

不引入 OpenTelemetry,用手写 span 结构 + JSON trace 日志:

- Tracer:进程内单例,维护当前 span 栈(线程/协程上下文经 contextvars 隔离)
- trace_span(name, **attrs):上下文管理器,span 完成时打一条 JSON trace 日志,
  含 trace_id / parent_id / duration_ms / attrs;异常自动标记 error + 异常信息
- trace_id:一条链路共享(请求级),与 request_id 同源或独立生成;span 嵌套形成树

零外部依赖:仅 stdlib contextvars / time / logging / uuid / json。
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from .logging_config import get_request_id

logger = logging.getLogger("obs.trace")

# 当前活动 span 栈:每个逻辑上下文(请求/协程)独立,嵌套的 trace_span 进栈出栈
_span_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "trace_span_stack", default=()
)

# 当前上下文绑定的 trace_id:链路根 span 生成,子 span 继承
_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


def get_trace_id() -> Optional[str]:
    """读取当前上下文的 trace_id(无则 None)。"""
    return _trace_id.get()


def set_trace_id(trace_id: Optional[str]) -> None:
    """设置当前上下文 trace_id(链路入口用;测试也可用)。"""
    _trace_id.set(trace_id)


def _new_trace_id() -> str:
    """trace_id 与 request_id 同源(有 request_id 时复用),否则独立生成。"""
    return get_request_id() or uuid.uuid4().hex[:16]


class Span:
    """单个 span:记录 start/end/耗时/attrs/error,可嵌套。"""

    __slots__ = (
        "name", "trace_id", "span_id", "parent_id", "start_ns",
        "end_ns", "attrs", "error", "error_msg",
    )

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_id: Optional[str],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.start_ns = time.perf_counter_ns()
        self.end_ns: Optional[int] = None
        self.attrs: Dict[str, Any] = dict(attrs or {})
        self.error: Optional[str] = None
        self.error_msg: Optional[str] = None

    def finish(self) -> None:
        """标记结束;未结束就被 finish 时以当前时间计。"""
        if self.end_ns is None:
            self.end_ns = time.perf_counter_ns()

    @property
    def duration_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return (end - self.start_ns) / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 3),
            "attrs": self.attrs,
        }
        if self.error:
            payload["error"] = self.error
            payload["error_msg"] = self.error_msg
        return payload


class Tracer:
    """span 生命周期管理:进栈/出栈,完成时打 JSON trace 日志。"""

    def __init__(self) -> None:
        self._logger = logger

    def _current_span(self) -> Optional[Span]:
        stack = _span_stack.get()
        return stack[-1] if stack else None

    def start_span(
        self, name: str, **attrs: Any
    ) -> Span:
        parent = self._current_span()
        trace_id = parent.trace_id if parent else _trace_id.get() or _new_trace_id()
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            attrs=attrs,
        )
        # 链路根 span 生成后把 trace_id 绑定到当前上下文,子 span 继承
        if parent is None:
            _trace_id.set(trace_id)
        _span_stack.set(_span_stack.get() + (span,))
        return span

    def end_span(
        self,
        span: Span,
        error: Optional[str] = None,
        error_msg: Optional[str] = None,
    ) -> None:
        span.finish()
        if error:
            span.error = error
            span.error_msg = error_msg
        # 出栈:只允许 pop 栈顶的 span(防御错序结束)
        stack = _span_stack.get()
        if stack and stack[-1] is span:
            _span_stack.set(stack[:-1])
        self._emit(span)

    def _emit(self, span: Span) -> None:
        """span 完成 → 打一条 JSON trace 日志(与 Logs 同 JSON 体系)。"""
        payload = span.to_dict()
        self._logger.info(
            "span_complete name=%s trace_id=%s span_id=%s duration_ms=%.1f error=%s",
            span.name, span.trace_id, span.span_id, span.duration_ms, span.error or "none",
            extra={"trace": payload},
        )

    def reset(self) -> None:
        """清空当前上下文 span 栈与 trace_id(测试隔离用)。"""
        _span_stack.set(())
        _trace_id.set(None)


tracer = Tracer()


class trace_span:
    """上下文管理器:with trace_span("rag_search", mode="hybrid"): ...

    进入时 start span,正常退出 end,异常退出自动标记 error(exception 类名 + 消息)。
    嵌套使用自动成为父子;同链路共享 trace_id。
    """

    def __init__(self, name: str, **attrs: Any) -> None:
        self.name = name
        self.attrs = attrs
        self._span: Optional[Span] = None

    def __enter__(self) -> Span:
        self._span = tracer.start_span(self.name, **self.attrs)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._span is None:
            return False
        if exc is not None:
            tracer.end_span(
                self._span,
                error=exc_type.__name__ if exc_type else "error",
                error_msg=str(exc),
            )
        else:
            tracer.end_span(self._span)
        return False
