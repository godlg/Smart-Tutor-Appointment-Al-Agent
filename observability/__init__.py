"""可观测性模块 (补4)

提供 Prometheus metrics 定义、FastAPI 中间件、业务计数器,
供 app.py 在 create_app() 时挂载, /metrics 端点暴露。

Logs + Traces (补4):
- logging_config: JSON 结构化日志 + request_id contextvars + RequestIdFilter
- request_id_middleware: 纯 ASGI 中间件,生成/透传 request_id,响应头带回
- tracing: 零依赖 span 工具(trace_span / Tracer),覆盖 RAG 检索 / Agent 路由 / 工具调用

设计要点:
- 全部 metric 单例 (模块级), 避免重复创建导致 /metrics 报 DUPLICATE
- HTTP label 用 method + path (而非完整 URL, 避免 high-cardinality)
- upload 用业务级 outcome label (success / skipped / rejected),
  而非 HTTP status (200/400 都可能是 success)
- span 输出走同一条 JSON 日志体系, trace_id 与 request_id 同源
"""
from .logging_config import (
    JsonFormatter,
    RequestIdFilter,
    get_request_id,
    set_request_id,
    setup_logging,
)
from .metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
    KNOWLEDGE_UPLOAD_TOTAL,
    KNOWLEDGE_UPLOAD_CHUNKS,
    KNOWLEDGE_UPLOAD_BYTES,
    KNOWLEDGE_SEARCH_TOTAL,
    register_metrics,
)
from .middleware import PrometheusMiddleware
from .request_id_middleware import RequestIdMiddleware
from .tracing import Tracer, Span, get_trace_id, set_trace_id, trace_span, tracer

__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION",
    "KNOWLEDGE_UPLOAD_TOTAL",
    "KNOWLEDGE_UPLOAD_CHUNKS",
    "KNOWLEDGE_UPLOAD_BYTES",
    "KNOWLEDGE_SEARCH_TOTAL",
    "PrometheusMiddleware",
    "register_metrics",
    # Logs + Traces (补4)
    "setup_logging",
    "JsonFormatter",
    "RequestIdFilter",
    "get_request_id",
    "set_request_id",
    "RequestIdMiddleware",
    "Tracer",
    "Span",
    "trace_span",
    "tracer",
    "get_trace_id",
    "set_trace_id",
]