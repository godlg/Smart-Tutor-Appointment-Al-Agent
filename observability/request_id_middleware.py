"""
Request-ID 中间件 (补4: Logs 的请求级关联)

纯 ASGI 中间件(与 PrometheusMiddleware 同风格,避开 BaseHTTPMiddleware 坑):
- 入口生成 request_id(uuid4 hex 短版)写入 contextvars
- 响应头 X-Request-ID 带回(透传客户端传来的值,或回显生成的)
- 所有日志经 RequestIdFilter 自动带上当前 request_id
"""
from __future__ import annotations

import uuid
from typing import Callable

from .logging_config import set_request_id

_X_REQUEST_ID = "x-request-id"


class RequestIdMiddleware:
    """为每个 HTTP 请求生成/透传 request_id,并注入响应头。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(_X_REQUEST_ID.encode())
        request_id = (
            incoming.decode() if incoming else uuid.uuid4().hex[:16]
        )
        set_request_id(request_id)

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = list(message["headers"])
                headers.append((_X_REQUEST_ID.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # 请求结束清掉上下文,避免污染后续请求(测试/长驻进程)
            set_request_id(None)
