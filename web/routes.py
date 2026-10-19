"""
Web 界面路由聚合器（家教场景）

分面说明：
- 家长面 → `web/parent_routes.py`（URL 与行为完全不变，无鉴权）
- 机构后台面 → `web/console_routes.py`（`/console/*` + 控制台鉴权）

本模块只做聚合，对外仍暴露同一个 `router`（`web/__init__.py` 与 `app.py` 不变）
以及聊天入口的既有名字，供既有调用方按原路径导入。
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .parent_routes import (
    ChatRequest,
    MAX_CHAT_ID_LENGTH,
    ProcessUserInput_stream,
    _chat_response,
    chat_endpoint,
    chat_stream_endpoint,
    router as parent_router,
)
from .console_routes import router as console_router
from .teacher_routes import router as teacher_router

router = APIRouter()
router.include_router(parent_router)
router.include_router(console_router)
router.include_router(teacher_router)

__all__ = [
    "router",
    "parent_router",
    "console_router",
    "ChatRequest",
    "MAX_CHAT_ID_LENGTH",
    "ProcessUserInput_stream",
    "StreamingResponse",
    "JSONResponse",
    "_chat_response",
    "chat_endpoint",
    "chat_stream_endpoint",
]
