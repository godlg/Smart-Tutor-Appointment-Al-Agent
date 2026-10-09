"""
通用异常处理（兜底 log）

只保留一个未处理异常的兜底 handler:
- log 完整 traceback
- 返回统一 JSONResponse(500)

原版本里的 BusinessException / api_exception_handler 已删除,
原因: BusinessException 在代码库中 0 次 raise, 且它是 HTTPException 子类,
注册到 FastAPI 不会真正生效(FastAPI 内置 handler 会先拦截).
"""
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器 (兜底)"""
    import traceback
    error_detail = f"未处理异常: {exc}"
    stack_trace = traceback.format_exc()
    logger.error(f"{error_detail}\n{stack_trace}")

    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误"},
    )