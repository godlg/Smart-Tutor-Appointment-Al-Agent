"""
FastAPI Prometheus 中间件 (补4)

记录每个请求的:
- 总数 (method + path + status)
- 延迟分布 (method + path)

为什么不用 BaseHTTPMiddleware:
- Starlette 的 BaseHTTPMiddleware 在 call_next 之前 scope["route"] 还是 None
  (route 是 FastAPI 路由器在内部 ASGI 调用之后才写入 scope 的)
- 所以我们用纯 ASGI 中间件,在 inner app 返回后从 scope 取 route template

path label 策略:
- 优先用 route.path (e.g. /api/knowledge/{knowledge_id})
- fallback 到 raw path (e.g. /metrics 或 404 的路径)
"""
import time
from typing import Any, Awaitable, Callable

from .metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION


class PrometheusMiddleware:
    """纯 ASGI 中间件 (不被 BaseHTTPMiddleware 的 route-lookup 时机坑)。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # 只处理 HTTP 请求 (lifespan / websocket 不打点)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        # /metrics 端点本身不打点(否则 Prometheus 抓取会无限累加)
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500  # default in case inner never sends response.start

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            # 此时 scope["route"] 已被 FastAPI 路由器写入
            route = scope.get("route")
            path_label = getattr(route, "path", None) or path

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path_label,
                status=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION.labels(
                method=method,
                path=path_label,
            ).observe(duration)