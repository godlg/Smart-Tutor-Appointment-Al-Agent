"""
家长面路由（家教场景）

面归属：本模块只承载**家长面**（对话入口 + 家长可见页面），
URL 与行为与分面之前逐字一致，**一律不加任何控制台鉴权**。
机构后台面在 `web/console_routes.py`，由 `web/routes.py` 聚合。
"""

import sys
from contextlib import aclosing
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from api.chat_handler import ProcessUserInput_stream
import logging

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="web/templates")
router = APIRouter(tags=["Web界面"])
MAX_CHAT_ID_LENGTH = 256

# `web/routes.py` 分面后只是聚合器，但 `web.routes.<name>` 仍是调用方既有的
# 读取/替换位置（测试通过 monkeypatch 替换聊天入口、禁用 StreamingResponse）。
_AGGREGATOR_MODULE = "web.routes"


def _aggregated(name: str, fallback):
    """优先取聚合器模块 `web.routes` 上的同名对象，取不到才用本模块的。

    调用时回查一次：默认行为不变（两边通常就是同一个对象），
    同时保住 `web.routes.ProcessUserInput_stream` 这类既有补丁点。
    """
    module = sys.modules.get(_AGGREGATOR_MODULE)
    if module is not None:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return fallback


class ChatRequest(BaseModel):
    message: str
    state: str | None = None
    # 会回传到 HTTP header，校验须早于身份签发和持久绑定；兼容前端 UUID。
    session_id: str | None = Field(
        default=None, min_length=1, max_length=MAX_CHAT_ID_LENGTH,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    # 只作匿名 owner 下的记忆 namespace，不证明真实学员归属。
    student_id: str | None = Field(default=None, min_length=1, max_length=MAX_CHAT_ID_LENGTH)


def _chat_response(chat: ChatRequest, request: Request) -> StreamingResponse | JSONResponse:
    from api.identity import resolve_chat_identity, attach_identity_cookie
    from config.identity import set_current_identity, reset_current_identity

    identity, error = resolve_chat_identity(
        request, chat.session_id, chat.student_id,
        student_supplied="student_id" in chat.model_fields_set,
    )
    if error is not None:
        attach_identity_cookie(error, identity, request)
        return error

    async def token_generator():
        token = set_current_identity(identity)
        try:
            # 保留既有 kwargs；无客户端 sid 时主控使用 ContextVar 中已绑定的 sid，
            # 而不是 legacy/global。历史 human/ai 只由主控记录，Web 不重复写入。
            stream_entry = _aggregated("ProcessUserInput_stream", ProcessUserInput_stream)
            async with aclosing(stream_entry(
                chat.message, session_id=chat.session_id
            )) as stream:
                async for part in stream:
                    yield part
        finally:
            reset_current_identity(token)

    response = _aggregated("StreamingResponse", StreamingResponse)(
        token_generator(), media_type="text/plain",
        headers={"X-Tutor-Session-Id": identity.session_id},
    )
    attach_identity_cookie(response, identity, request)
    return response


@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/chat/stream", summary="流式聊天")
async def chat_stream_endpoint(chat: ChatRequest, request: Request):
    # 同步SQLite事务/锁等待在线程池完成，不阻塞其他聊天流。
    return await run_in_threadpool(_chat_response, chat, request)


@router.post("/chat", summary="兼容性聊天接口")
async def chat_endpoint(chat: ChatRequest, request: Request):
    # 同步SQLite事务/锁等待在线程池完成，不阻塞其他聊天流。
    return await run_in_threadpool(_chat_response, chat, request)


# =========================================
# 家教核心页面
# =========================================

@router.get("/teachers", response_class=HTMLResponse, summary="老师库")
async def teacher_list_page(request: Request):
    """老师库列表页"""
    try:
        from services.teacher_service import TeacherService
        svc = TeacherService()
        svc.initialize_default_teachers()
        teachers = svc.get_all_teachers()
        subjects = svc.get_all_subjects()
        grade_levels = svc.get_all_grade_levels()
        return templates.TemplateResponse("teacher_list.html", {
            "request": request, "teachers": teachers,
            "subjects": subjects, "grade_levels": grade_levels,
        })
    except Exception as e:
        logger.error(f"加载老师库失败: {e}")
        return templates.TemplateResponse("teacher_list.html", {
            "request": request, "teachers": [], "subjects": [], "grade_levels": [],
            "error": str(e),
        })


@router.get("/teachers/{teacher_id}", response_class=HTMLResponse, summary="老师详情页")
async def teacher_detail_page(request: Request, teacher_id: int):
    """老师详情 + 预约试讲"""
    return templates.TemplateResponse("teacher_detail.html", {"request": request})


@router.get("/packages", response_class=HTMLResponse, summary="课时套餐")
async def packages_page(request: Request):
    """套餐列表页"""
    try:
        from services.package_service import PackageService
        svc = PackageService()
        svc.initialize_default_packages()
        packages = svc.get_all_packages()
        return templates.TemplateResponse("packages.html", {
            "request": request, "packages": packages,
        })
    except Exception as e:
        logger.error(f"加载套餐失败: {e}")
        return templates.TemplateResponse("packages.html", {
            "request": request, "packages": [], "error": str(e),
        })


@router.get("/schedule-calendar", response_class=HTMLResponse, summary="排课日历")
async def schedule_calendar_page(request: Request):
    """排课日历"""
    return templates.TemplateResponse("schedule_calendar.html", {"request": request})


@router.get("/lesson-series", response_class=HTMLResponse, summary="我的系列课")
async def lesson_series_page(request: Request):
    """系列课列表页"""
    return templates.TemplateResponse("lesson_series.html", {"request": request})


@router.get("/progress-dashboard", response_class=HTMLResponse, summary="学习进度看板")
async def progress_dashboard_page(request: Request):
    """学习进度看板"""
    return templates.TemplateResponse("progress_dashboard.html", {"request": request})


@router.get("/student-profile", response_class=HTMLResponse, summary="学生档案")
async def student_profile_page(request: Request):
    """学生档案"""
    return templates.TemplateResponse("student_profile.html", {"request": request})


@router.get("/parent-center", response_class=HTMLResponse, summary="家长中心")
async def parent_center_page(request: Request):
    """家长中心"""
    return templates.TemplateResponse("parent_center.html", {"request": request})
