"""
老师面路由（教师端最小闭环）

面归属：本模块只承载**老师面**，挂在 `/teacher` 前缀下。
- `/teacher/login`：**公开**，登录复用 `POST /api/console/login`（同一账号体系）
- `/teacher`：`require_teacher()`，未登录 → 401，非老师账号 → 403

老师面不接受控制台角色；`/console/*` 对 teacher 账号一律 403。
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.core.console_auth import require_teacher

templates = Jinja2Templates(directory="web/templates")
router = APIRouter(prefix="/teacher", tags=["老师面"])


@router.get("/login", response_class=HTMLResponse, summary="老师登录页")
async def teacher_login_page(request: Request):
    """登录页必须公开：否则老师没有入口。"""
    return templates.TemplateResponse("teacher_login.html", {"request": request})


@router.get("", response_class=HTMLResponse, summary="老师待办首页")
async def teacher_home_page(request: Request, teacher: dict = Depends(require_teacher())):
    """老师待办页：页面数据由前端调用 /api/teacher/* 获取。"""
    return templates.TemplateResponse("teacher_home.html", {"request": request})
