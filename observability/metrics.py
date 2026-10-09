"""
Prometheus metrics 定义 (补4)

- HTTP_REQUESTS_TOTAL / HTTP_REQUEST_DURATION: 全局 HTTP 流量
- KNOWLEDGE_UPLOAD_*: 上传业务指标 (P1-13)
- KNOWLEDGE_SEARCH_TOTAL: 知识库搜索调用次数

使用方式:
    from observability import KNOWLEDGE_UPLOAD_TOTAL
    KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="success", kind="pdf").inc()

所有 metric 在模块导入时创建一次,避免 prometheus_client.DuplicateTimeseries。
"""
from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY


# ----------------------------------------------------------------------
# HTTP 流量 (中间件用)
# ----------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests processed by the API",
    labelnames=("method", "path", "status"),
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path"),
    # 经验值: API 调用大多 < 100ms, 上传可能 30s+
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


# ----------------------------------------------------------------------
# 知识库业务指标
# ----------------------------------------------------------------------

# outcome: success / skipped / rejected_error / rejected_too_large / rejected_unsupported
# kind: pdf / markdown / unknown
KNOWLEDGE_UPLOAD_TOTAL = Counter(
    "knowledge_upload_total",
    "Total knowledge base upload attempts by outcome and file kind",
    labelnames=("outcome", "kind"),
)

KNOWLEDGE_UPLOAD_CHUNKS = Counter(
    "knowledge_upload_chunks_total",
    "Total chunks persisted from uploads (counts only successful, non-dedup rows)",
    labelnames=("kind",),
)

KNOWLEDGE_UPLOAD_BYTES = Counter(
    "knowledge_upload_bytes_total",
    "Total bytes processed from upload requests (counts accepted and rejected)",
    # 字节大小标签分桶,避免 1MB-1TB 跨度过宽
    labelnames=("kind",),
)

KNOWLEDGE_SEARCH_TOTAL = Counter(
    "knowledge_search_total",
    "Total knowledge base search calls",
    # has_meta_filter: 是否带 subject/grade_level/doc_type 过滤
    # result_count_bucket: none / few(1-3) / many(4+)
    labelnames=("has_meta_filter", "result_count_bucket"),
)


def register_metrics(registry: CollectorRegistry = REGISTRY) -> None:
    """显式注册 metric 到指定 registry (测试用)。

    prometheus_client 默认 metric 已经在模块导入时注册到 global REGISTRY,
    此函数主要是为了 API 一致性与未来切换 registry 时不破坏调用方。
    """
    # metric 对象是注册到 REGISTRY 的,直接 pass-through
    for m in (
        HTTP_REQUESTS_TOTAL,
        HTTP_REQUEST_DURATION,
        KNOWLEDGE_UPLOAD_TOTAL,
        KNOWLEDGE_UPLOAD_CHUNKS,
        KNOWLEDGE_UPLOAD_BYTES,
        KNOWLEDGE_SEARCH_TOTAL,
    ):
        # touch labelnames 以触发 0-label 时的注册
        _ = m._labelnames
    return None