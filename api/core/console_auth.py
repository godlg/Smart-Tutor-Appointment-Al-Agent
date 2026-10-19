"""控制台鉴权依赖(机构后台分面 A)

只提供可挂载的 FastAPI 依赖,不碰任何现有家长侧路由:
- `require_console_user(...)` 返回依赖函数,从控制台 cookie 解析会话,
  未登录 → 401,角色不匹配 → 403,通过则**返回 user dict** 供路由注入。
- `require_teacher()` 老师面闸门:未登录 → 401;非 teacher 角色或无 teacher_id → 403。
  teacher 角色**不属于** `ALL_ROLES`,不得加入任何控制台/知识库/接管的允许集合。

边界(勿夸大):控制台会话只证明「机构后台某个账号登录过」,
不代表任何家长身份,也不代表真实业务数据归属。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request

from config.console_auth import CONSOLE_COOKIE_NAME

# 控制台角色常量(与 db.models.ConsoleUser.role 取值一致)
ROLE_ADMIN = "admin"
ROLE_SUPPORT = "support"
ROLE_ACADEMIC = "academic"
ROLE_TEACHER = "teacher"  # 老师面专用;不在 ALL_ROLES 内

ALL_ROLES = (ROLE_ADMIN, ROLE_SUPPORT, ROLE_ACADEMIC)

_console_auth_service = None


def get_console_auth_service():
    """懒加载控制台鉴权服务(单例);测试可注入指向临时库的实例。"""
    global _console_auth_service
    if _console_auth_service is None:
        from services.console_auth_service import ConsoleAuthService
        _console_auth_service = ConsoleAuthService()
    return _console_auth_service


def _current_user(request: Request) -> Dict[str, Any]:
    """从 cookie 解析会话;未登录 → 401。"""
    service = get_console_auth_service()
    token: Optional[str] = request.cookies.get(CONSOLE_COOKIE_NAME)
    user = service.resolve(token)
    if not user:
        raise HTTPException(status_code=401, detail="需要控制台登录")
    return user


def require_console_user(*roles: str):
    """构造控制台鉴权依赖。

    参数 `roles` 为空 → 只要求「已登录任一控制台角色」;
    非空 → 要求当前会话角色在允许集合内,否则 403。
    """

    def _dependency(request: Request) -> Dict[str, Any]:
        user = _current_user(request)
        if roles and user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="当前角色无权访问该功能")
        return user

    return _dependency


def require_teacher() -> Callable[[Request], Dict[str, Any]]:
    """老师面闸门:角色必须是 teacher 且绑定了 teacher_id,通过则返回 user dict(含 teacher_id)。"""

    def _dependency(request: Request) -> Dict[str, Any]:
        user = _current_user(request)
        if user.get("role") != ROLE_TEACHER or user.get("teacher_id") is None:
            raise HTTPException(status_code=403, detail="仅老师账号可访问")
        return user

    return _dependency
