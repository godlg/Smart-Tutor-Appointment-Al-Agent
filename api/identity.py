"""匿名 Web 身份接线：持久 cookie、会话归属和学生记忆 namespace。

无登录系统；服务端认可的匿名身份不能证明真实家长或业务学员归属。
请求体 user_id 不作为授权，student_id 只用于该匿名 owner 的记忆子空间。
"""

import os
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from config.identity import (
    IDENTITY_COOKIE_NAME,
    Identity,
    get_current_identity,
    reset_current_identity,
    set_current_identity,  # noqa: F401 (保留原有 re-export)
)

IDENTITY_COOKIE_MAX_AGE_SECONDS = 365 * 24 * 60 * 60
_identity_service = None


def get_identity_service():
    """懒加载服务；测试可注入真实临时库 service。"""
    global _identity_service
    if _identity_service is None:
        from services.identity_service import IdentityService
        _identity_service = IdentityService()
    return _identity_service


def resolve_request_identity(request: Request) -> Identity:
    """只有持久登记过的 cookie 才复用；缺失或伪造 cookie 重新签发。"""
    service = get_identity_service()
    return service.resolve_identity(request.cookies.get(IDENTITY_COOKIE_NAME))


def resolve_chat_identity(
    request: Request,
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
    *,
    student_supplied: bool = False,
) -> tuple[Identity, Optional[JSONResponse]]:
    """在流开始前生成/绑定会话，并恢复同 owner 的既有 student scope。

    预读仅用于请求省略 student 时恢复 namespace，不能代替原子归属校验。
    显式 null 也是 scope 选择，不能用它清除既有 student 绑定。
    不查询业务 student 表，也不把 namespace 认领视作真实业务认证。
    """
    service = get_identity_service()
    identity = resolve_request_identity(request)
    if session_id and student_id is None and not student_supplied:
        binding = service.repo.get_session(session_id)
        if binding is not None and binding["user_id"] == identity.user_id:
            student_id = binding["student_id"]
    identity = Identity(
        user_id=identity.user_id,
        student_id=student_id,
        session_id=session_id or service.new_session_id(),
    )
    error = enforce_session_ownership(identity, identity.session_id)
    return identity, error


def attach_identity_cookie(
    response: Response, identity: Identity, request: Optional[Request] = None
) -> None:
    """一年期 HttpOnly cookie；HTTPS 默认 Secure，代理可配置强制开启。

    TUTOR_IDENTITY_COOKIE_SECURE=true/1/yes/on 可用于 HTTPS 终止代理后方。
    不直接信任客户端转发头；配置 false 也不降级 HTTPS 请求的 Secure。
    """
    force_secure = os.getenv("TUTOR_IDENTITY_COOKIE_SECURE", "").strip().lower()
    is_https = request is not None and request.url.scheme == "https"
    response.set_cookie(
        key=IDENTITY_COOKIE_NAME,
        value=identity.user_id,
        max_age=IDENTITY_COOKIE_MAX_AGE_SECONDS,
        expires=IDENTITY_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=is_https or force_secure in {"1", "true", "yes", "on"},
        samesite="lax",
        path="/",
    )


def enforce_session_ownership(
    identity: Identity, session_id: Optional[str]
) -> Optional[JSONResponse]:
    """调用持久库原子绑定/校验；403/409 必须在 StreamingResponse 前返回。"""
    if not session_id:
        return None
    service = get_identity_service()
    try:
        service.assert_session_owned(identity, session_id)
    except (service.SessionOwnershipError, service.StudentClaimError) as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
    return None
