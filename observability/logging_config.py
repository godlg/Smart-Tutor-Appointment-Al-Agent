"""
结构化日志配置 (补4: Logs)

统一 logging 体系:JSON 行格式(ts/level/logger/msg/request_id/extra),
请求级 request_id 由中间件生成写入 contextvars,日志 Filter 自动附加。

- setup_logging(level, json=True): 替换 root handler 为 JSON formatter
- request_id_context: contextvars 存当前请求的 request_id(可透传 header)
- RequestIdFilter: logging.Filter,自动给每条日志附 request_id

不破坏现有 print/logger 调用:只替换 formatter/handler,现有 logger.info/warning 照常工作,
只是输出变成 JSON 行。外部 collector / 日志文件留给部署层,这里只保证 console 干净。
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any, Dict, Optional

from config.settings import settings

# contextvars 存当前请求的 request_id:每个请求(或后台任务)独立上下文,
# 中间件在入口 set,日志 Filter 在 emit 时读。线程/协程安全。
request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """读取当前上下文里的 request_id(没有则 None)。"""
    return request_id_context.get()


def set_request_id(request_id: Optional[str]) -> None:
    """设置当前上下文的 request_id。"""
    request_id_context.set(request_id)


class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化成单行 JSON。

    字段: ts / level / logger / msg / request_id / extra(自动展开 message 里
    未占用的 kwargs,如 logger.info("x", user=1) → extra.user=1)
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        # extra 字段:message kwargs(如 logger.info("x", stages=...))
        # 只取 __dict__ 里用户显式传的,不污染标准 LogRecord 属性
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "taskName",
                "message", "asctime", "request_id",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:  # 极端情况(不可序列化)回退普通格式
            return super().format(record)


class DynamicStreamHandler(logging.StreamHandler):
    """每次 emit 动态读 sys.stdout:兼容 pytest capsys / 日志重定向。

    普通 StreamHandler 在构造时绑定 stream,而 pytest 会替换并关闭原 stdout,
    导致 'I/O operation on closed file'。这里 emit 时取当前 sys.stdout。
    """

    def __init__(self) -> None:
        # 不传 stream:父类会用 sys.stderr 兜底,但我们在 emit 里强制覆盖
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass  # 忽略外部赋值,永远跟随当前 sys.stdout


class RequestIdFilter(logging.Filter):
    """日志过滤器:把 contextvars 里的 request_id 塞进 record。

    直接附加到 handler 上,所有 logger 都会经过;没有 request_id 时 record 不增加字段,
    JsonFormatter 统一输出 null。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging(
    level: Optional[str] = None,
    json: bool = True,
    force: bool = False,
) -> None:
    """统一配置 root logger:console handler + (可选) JSON 格式。

    Args:
        level: 日志级别字符串,默认取 env LOG_LEVEL(未设置则 INFO)。
        json: True 输出 JSON 行, False 用可读的普通格式。
        force: True 时清掉 root 现有 handler 再重建(幂等;应用启动用一次即可)。

    幂等:重复调用不会堆 handler(默认不 force 时只追加一次)。
    """
    level = (level or settings.log_level).upper()
    root = logging.getLogger()

    # 幂等:已有我们装的 handler 就不再重复添加
    for h in root.handlers:
        if getattr(h, "_obs_json_handler", False):
            if root.level > getattr(logging, level, logging.INFO):
                root.setLevel(getattr(logging, level, logging.INFO))
            return

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    # 用 DynamicStreamHandler:emit 时动态读当前 sys.stdout(兼容 pytest capsys 替换)
    handler = DynamicStreamHandler()
    handler._obs_json_handler = True  # type: ignore[attr-defined]
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
