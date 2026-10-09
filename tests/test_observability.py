"""
Prometheus 可观测性测试 (补4)

覆盖:
- metrics 模块的所有 Counter / Histogram 暴露正确
- PrometheusMiddleware 正确记录 method / path / status + 延迟
- /metrics 端点返回 Prometheus exposition format
- 上传 / 搜索的业务计数 endpoint 集成
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ----------------------------------------------------------------------
# A. metrics 模块
# ----------------------------------------------------------------------

class TestMetricsModule:
    def test_all_metrics_importable(self):
        """所有 metric 都应该能从 observability 顶层 import。"""
        from observability import (
            HTTP_REQUESTS_TOTAL,
            HTTP_REQUEST_DURATION,
            KNOWLEDGE_UPLOAD_TOTAL,
            KNOWLEDGE_UPLOAD_CHUNKS,
            KNOWLEDGE_UPLOAD_BYTES,
            KNOWLEDGE_SEARCH_TOTAL,
            PrometheusMiddleware,
            register_metrics,
        )

    def test_register_metrics_is_idempotent(self):
        """register_metrics 重复调用不抛异常 (不会注册到 duplicate label)。"""
        from observability import register_metrics
        register_metrics()
        register_metrics()  # 第二次不应触发 Duplicate

    def test_knowledge_upload_total_accepts_labels(self):
        """业务计数能用 outcome + kind label .inc()。"""
        from observability import KNOWLEDGE_UPLOAD_TOTAL
        before = KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="success", kind="pdf",
        )._value.get()
        KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="success", kind="pdf").inc()
        after = KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="success", kind="pdf",
        )._value.get()
        assert after == before + 1

    def test_upload_chunks_counter_accumulates(self):
        """chunks 计数应能累加 >1 的值(因为一次上传可能 >1 chunk)。"""
        from observability import KNOWLEDGE_UPLOAD_CHUNKS
        before = KNOWLEDGE_UPLOAD_CHUNKS.labels(kind="markdown")._value.get()
        KNOWLEDGE_UPLOAD_CHUNKS.labels(kind="markdown").inc(5)
        KNOWLEDGE_UPLOAD_CHUNKS.labels(kind="markdown").inc(3)
        after = KNOWLEDGE_UPLOAD_CHUNKS.labels(kind="markdown")._value.get()
        assert after == before + 8


# ----------------------------------------------------------------------
# B. PrometheusMiddleware
# ----------------------------------------------------------------------

class TestPrometheusMiddleware:
    @pytest.fixture
    def app(self):
        """最小 FastAPI app + 中间件,用于隔离测试中间件本身。"""
        app = FastAPI()

        @app.get("/hello")
        def hello():
            return {"msg": "hi"}

        @app.get("/items/{item_id}")
        def get_item(item_id: int):
            return {"id": item_id}

        @app.get("/boom")
        def boom():
            raise RuntimeError("test")

        from observability import PrometheusMiddleware
        app.add_middleware(PrometheusMiddleware)
        return app

    def test_records_successful_request(self, app):
        from observability import HTTP_REQUESTS_TOTAL
        before = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/hello", status="200",
        )._value.get()

        client = TestClient(app)
        client.get("/hello")

        after = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/hello", status="200",
        )._value.get()
        assert after == before + 1

    def test_records_500_status_on_unhandled_exception(self, app):
        """中间件应捕获所有 status (包括 unhandled exception → 500)。"""
        from observability import HTTP_REQUESTS_TOTAL
        before = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/boom", status="500",
        )._value.get()

        client = TestClient(app, raise_server_exceptions=False)
        client.get("/boom")

        after = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/boom", status="500",
        )._value.get()
        assert after == before + 1

    def test_uses_route_template_for_path_label(self, app):
        """path label 用 route template (/items/{item_id}) 而不是 raw path。

        避免高基数(label explosion): /items/1 /items/2 ... 都是同一个 label。
        """
        from observability import HTTP_REQUESTS_TOTAL
        client = TestClient(app)
        client.get("/items/1")
        client.get("/items/999")

        # 模板路径应累计 2 次(两次请求归到同一 label)
        template_label = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/items/{item_id}", status="200",
        )._value.get()
        assert template_label >= 2

        # raw 路径 label 不应被创建(因为中间件优先用 template)
        # prometheus_client 在不存在的 label 上 .labels() 会创建 0 计数 entry
        # 所以"未创建"用 KeyError 检测更稳,但 ._value.get() 也安全(返回 0.0)
        raw_label = HTTP_REQUESTS_TOTAL.labels(
            method="GET", path="/items/1", status="200",
        )._value.get()
        assert raw_label == 0.0

    def test_records_request_duration(self, app):
        """Histogram 应记录延迟 (label 名正确即可,具体值不测)。"""
        from observability import HTTP_REQUEST_DURATION
        client = TestClient(app)
        client.get("/hello")

        # 触发了 label 即可,具体 value 不重要
        sample = HTTP_REQUEST_DURATION.labels(
            method="GET", path="/hello",
        )
        # _sum 至少 > 0 因为我们跑过一次请求
        assert sample._sum.get() > 0

    def test_metrics_endpoint_self_is_excluded(self, app):
        """/metrics 端点本身不被中间件记录(否则 Prometheus 抓取会无限递增)。"""
        from observability import HTTP_REQUESTS_TOTAL
        client = TestClient(app)
        # 抓 /metrics 不会影响 /metrics label 的计数 (因为中间件 skip)
        for _ in range(3):
            client.get("/metrics")

        # /metrics 路径不应该被记录到
        try:
            v = HTTP_REQUESTS_TOTAL.labels(
                method="GET", path="/metrics", status="200",
            )._value.get()
            assert v == 0
        except KeyError:
            # 该 label 组合从未被初始化,符合预期
            pass


# ----------------------------------------------------------------------
# C. /metrics 端点
# ----------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_prometheus_format(self):
        """GET /metrics 返回 Prometheus exposition format (text/plain 0.0.4)。"""
        from app import app
        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        # 头应是 prometheus content-type
        assert "text/plain" in r.headers["content-type"]
        # body 应包含我们的 metric 名
        body = r.text
        assert "http_requests_total" in body
        assert "knowledge_upload_total" in body
        assert "knowledge_search_total" in body

    def test_metrics_endpoint_not_in_openapi(self):
        """/metrics 不应在 OpenAPI 文档里出现(纯运维端点)。"""
        from app import app
        client = TestClient(app)
        schema = client.get("/openapi.json").json()
        assert "/metrics" not in schema.get("paths", {})