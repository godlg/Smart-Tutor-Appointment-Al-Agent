"""
API 端点装饰器（消除 39 处重复 try/except 模板）

原模式:
    @router.post(...)
    async def foo(req):
        try:
            ...
            return XxxResponse(...)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

新模式:
    @router.post(...)
    @api_endpoint
    async def foo(req):
        ...
        return XxxResponse(...)

装饰器自动:
- 捕获 HTTPException 让其原样向上抛(避免遮盖 4xx 状态码)
- 其它异常统一转为 HTTPException(500, str(e))
- 可选 record_behavior: 用 @api_endpoint(action_type="...") 自动记录行为
"""
import functools
import logging
from typing import Optional, Callable, Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def api_endpoint(
    func: Optional[Callable] = None,
    *,
    action_type: Optional[str] = None,
    teacher_id_from: Optional[str] = None,
):
    """
    统一 API 端点包装。

    用法 1 (纯错误包装):
        @router.post(...)
        @api_endpoint
        async def foo(req): ...

    用法 2 (同时记录行为):
        @router.post(...)
        @api_endpoint(action_type="trial_book", teacher_id_from="request.teacher_id")
        async def book(req: TrialBookingRequest):
            ...
            # 不需要再手动 record_behavior
    """
    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"API {fn.__name__} 未处理异常")
                raise HTTPException(status_code=500, detail=f"{fn.__name__} 失败: {e}")

            # 主业务成功 -> 记录行为(失败不阻塞)
            if action_type:
                _record_behavior_safe(
                    action_type=action_type,
                    teacher_id=_resolve(teacher_id_from, args, kwargs),
                    action_data=_extract_action_data(args, kwargs),
                )
            return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"API {fn.__name__} 未处理异常")
                raise HTTPException(status_code=500, detail=f"{fn.__name__} 失败: {e}")

            if action_type:
                _record_behavior_safe(
                    action_type=action_type,
                    teacher_id=_resolve(teacher_id_from, args, kwargs),
                    action_data=_extract_action_data(args, kwargs),
                )
            return result

        # FastAPI 路由都是 async
        if asyncio_iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    # 允许 @api_endpoint 和 @api_endpoint(...) 两种用法
    if func is not None and callable(func):
        return wrap(func)
    return wrap


def asyncio_iscoroutinefunction(fn: Callable) -> bool:
    """检查是否为 async 函数(避免在模块顶部 import asyncio)"""
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _resolve(path: Optional[str], args: tuple, kwargs: dict) -> Optional[int]:
    """从参数中按 dotted path 拿 teacher_id, 如 'request.teacher_id' 或 'lesson.teacher_id'"""
    if not path:
        return None
    cur: Any = None
    parts = path.split(".")
    # 优先 kwargs, 然后按位置参数 args[0]
    if parts[0] in kwargs:
        cur = kwargs[parts[0]]
    elif args:
        cur = args[0]
    else:
        return None
    for p in parts[1:]:
        if cur is None:
            return None
        cur = getattr(cur, p, None)
    return cur


def _extract_action_data(args: tuple, kwargs: dict) -> dict:
    """从 request 拿 action_data, 默认给空 dict"""
    req = kwargs.get("request") or (args[0] if args else None)
    if req is None:
        return {}
    data = {}
    for attr in ("subject", "duration_minutes", "rating", "would_continue",
                 "lesson_id", "feedback_text"):
        if hasattr(req, attr):
            data[attr] = getattr(req, attr)
    return data


def _record_behavior_safe(
    action_type: str,
    teacher_id: Optional[int] = None,
    action_data: Optional[dict] = None,
    user_id: str = "default_user",
) -> None:
    """统一封装 UserBehaviorService.record_behavior 调用。
    任何异常都被吞掉, 不影响主业务流."""
    try:
        from services.user_behavior_service import UserBehaviorService
        UserBehaviorService().record_behavior(
            user_id=user_id,
            action_type=action_type,
            action_data=action_data,
            teacher_id=teacher_id,
        )
    except Exception as e:
        logger.warning(f"行为记录失败({action_type}): {e}")