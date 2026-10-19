"""控制台鉴权 API(机构后台分面 A)

- POST /api/console/login   账号口令登录,签发控制台会话 cookie
- POST /api/console/logout  注销当前会话(幂等)
- GET  /api/console/me      返回当前登录账号的非敏感字段

边界(勿夸大):这是机构后台自己的账号体系,与家长侧匿名身份无关;
响应体不回传口令哈希等敏感字段。
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from config.console_auth import CONSOLE_COOKIE_NAME

from .core.console_auth import require_console_user
from .core.response_models import DataResponse

router = APIRouter(prefix="/api/console", tags=["控制台鉴权"])

_SECURE_FLAGS = {"1", "true", "yes", "on"}


class ConsoleLoginRequest(BaseModel):
    """控制台登录请求体。"""
    username: str
    password: str


def _is_secure_request(request: Request) -> bool:
    """HTTPS 直接 Secure;HTTPS 终止代理后方可用 TUTOR_IDENTITY_COOKIE_SECURE 强制。"""
    force_secure = os.getenv("TUTOR_IDENTITY_COOKIE_SECURE", "").strip().lower()
    return request.url.scheme == "https" or force_secure in _SECURE_FLAGS


@router.post("/login", response_model=DataResponse, summary="控制台登录")
async def console_login(payload: ConsoleLoginRequest, request: Request, response: Response):
    """校验口令并签发会话 cookie;失败一律 401,不区分具体原因。"""
    from services.console_auth_service import AuthError, ConsoleAuthService

    service = ConsoleAuthService()
    try:
        result = service.login(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    finally:
        service.close()

    response.set_cookie(
        key=CONSOLE_COOKIE_NAME,
        value=result["token"],
        httponly=True,
        secure=_is_secure_request(request),
        samesite="lax",
        path="/",
    )
    # 只回显 token 与本账号非敏感字段,不回显口令/哈希
    return DataResponse(
        message="登录成功",
        data={
            "token": result["token"],
            "username": result["username"],
            "role": result["role"],
            "display_name": result["display_name"],
            "expires_at": result["expires_at"],
        },
    )


@router.post("/logout", response_model=DataResponse, summary="控制台注销")
async def console_logout(request: Request, response: Response):
    """删除当前 cookie 对应会话并清 cookie;未登录调用同样返回 200(幂等)。"""
    from services.console_auth_service import ConsoleAuthService

    token = request.cookies.get(CONSOLE_COOKIE_NAME)
    service = ConsoleAuthService()
    try:
        service.logout(token)
    finally:
        service.close()

    response.delete_cookie(key=CONSOLE_COOKIE_NAME, path="/")
    return DataResponse(message="已注销", data=None)


@router.get("/me", response_model=DataResponse, summary="当前控制台账号")
async def console_me(user: dict = Depends(require_console_user())):
    """返回当前登录账号的非敏感字段(绝不返回 password_hash)。"""
    return DataResponse(
        message="获取当前账号成功",
        data={
            "username": user["username"],
            "role": user["role"],
            "display_name": user["display_name"],
        },
    )
