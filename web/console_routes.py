"""
机构后台面路由（控制台）

面归属：本模块只承载**机构后台面**，全部挂在 `/console` 前缀下并校验控制台会话。
家长面在 `web/parent_routes.py`，两者由 `web/routes.py` 聚合成同一个 `router`。

鉴权口径（与 `api/knowledge.py` / `api/handoff.py` 的闸门一致）：
- 知识库 / 行为分析：`academic` 或 `admin`
- 人工接管队列：`support` 或 `admin`
- 控制台首页：admin / support / academic（teacher 账号不得进入）
- `/console/login`：**公开**（未登录也必须能打开）

未登录 → 401，角色不匹配 → 403，均由依赖抛出，页面层不做 HTML 跳转特例。
"""

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.core.console_auth import (
    ALL_ROLES,
    ROLE_ACADEMIC,
    ROLE_ADMIN,
    ROLE_SUPPORT,
    require_console_user,
)

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="web/templates")
router = APIRouter(prefix="/console", tags=["机构后台"])

# 页面口径的角色矩阵：与同名 API 的闸门保持同一份定义，避免两处漂移
KNOWLEDGE_ROLES = (ROLE_ACADEMIC, ROLE_ADMIN)
SUPPORT_ROLES = (ROLE_SUPPORT, ROLE_ADMIN)
HOME_ROLES = ALL_ROLES  # 首页：三种机构角色；teacher 不在内


@router.get("/login", response_class=HTMLResponse, summary="控制台登录页")
async def console_login_page(request: Request):
    """登录页必须公开：否则未登录用户没有入口。"""
    return templates.TemplateResponse("console_login.html", {"request": request})


@router.get("", response_class=HTMLResponse, summary="控制台首页")
async def console_home_page(request: Request, user: dict = Depends(require_console_user(*HOME_ROLES))):
    """后台首页：admin / support / academic；老师账号 → 403（老师面见 /teacher）。"""
    return templates.TemplateResponse("console_home.html", {"request": request})


@router.get("/human-handoff", response_class=HTMLResponse, summary="人工接管队列")
async def human_handoff_page(
    request: Request, user: dict = Depends(require_console_user(*SUPPORT_ROLES))
):
    """人工接管队列（客服侧，前端 fetch /api/handoffs）。"""
    return templates.TemplateResponse("human_handoff.html", {
        "request": request, "title": "人工接管队列",
    })


@router.get("/knowledge", response_class=HTMLResponse, summary="知识库管理")
async def knowledge_page(
    request: Request, user: dict = Depends(require_console_user(*KNOWLEDGE_ROLES))
):
    """知识库管理页（直接调用 API 函数取数据，不重复走 HTTP 鉴权）。"""
    try:
        from api.knowledge import get_all_knowledge
        knowledge_data = await get_all_knowledge()
        documents = knowledge_data.get("documents", [])
        categories = knowledge_data.get("categories", [])
        return templates.TemplateResponse("knowledge_management.html", {
            "request": request, "documents": documents, "categories": categories,
        })
    except Exception as e:
        return templates.TemplateResponse("knowledge_management.html", {
            "request": request, "documents": [], "categories": [], "error": str(e),
        })


@router.get("/user_behavior_analysis", response_class=HTMLResponse, summary="家长行为分析")
async def user_behavior_analysis_page(
    request: Request, user: dict = Depends(require_console_user(*KNOWLEDGE_ROLES))
):
    return templates.TemplateResponse("user_behavior_analysis.html", {"request": request})
