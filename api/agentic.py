"""Agentic 工具调用演示 API（独立入口，不碰现有 /chat 状态机）。

`POST /agentic/chat`：
  - 入参 {message, teacher_id?, time_iso?, subject?}
  - time_iso 缺失时用 time_config.parse_datetime 从 message 里解析（最小实现）
  - 调 run_agentic_trial_booking 跑完整工具循环，返回 {"tool_called", "lesson_id", "message"}
  - 支持 AGENTIC_DB_PATH 环境变量把服务切到临时 DB（8003 隔离验证用），默认生产 DB。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/agentic", tags=["Agentic 工具调用"])


class AgenticChatRequest(BaseModel):
    message: str
    teacher_id: Optional[int] = None
    time_iso: Optional[str] = None
    subject: Optional[str] = None


class AgenticChatResponse(BaseModel):
    tool_called: bool
    lesson_id: Optional[int] = None
    message: str


def _build_services() -> Tuple[Any, Any]:
    """构造 (lesson_service, student_service)。

    AGENTIC_DB_PATH 设置后指向临时 DB（DatabaseRouter + 把服务 .db 重绑）；
    否则用真实服务（生产 DB）。不修改任何现有模块行为。
    """
    db_path = os.environ.get("AGENTIC_DB_PATH")

    from services.lesson_service import LessonService
    from services.student_profile_service import StudentProfileService

    ls = LessonService()
    ss = StudentProfileService()
    if db_path:
        from db.db_router import DatabaseRouter

        db = DatabaseRouter(db_path)
        ls.db = db
        ss.db = db
    return ls, ss


@router.post("/chat", response_model=AgenticChatResponse, summary="Agentic 试讲预约（工具调用演示）")
def agentic_chat(request: AgenticChatRequest) -> Dict[str, Any]:
    """
    演示 Agentic 工具调用闭环：

    - 带绝对时间来 → LLM 自主调用 book_trial_lesson → 真实落库，返回预约成功消息；
    - 带冲突时间来 → tool 返回 ERROR → LLM 生成换时间引导；
    - 缺参 → tool_called=False，返回回退消息。
    """
    # 最小实现：time_iso 缺失时尝试用 time_config.parse_datetime 解析 message 里的时间
    time_iso = request.time_iso
    if not time_iso:
        from config.time_config import time_config

        dt = time_config.parse_datetime(request.message)
        if dt is not None:
            time_iso = dt.strftime("%Y-%m-%d %H:%M")

    from config.model_provider import create_chat_model
    from agents.agentic.agentic_trial_booking import run_agentic_trial_booking

    lesson_service, student_service = _build_services()
    llm = create_chat_model()

    result = run_agentic_trial_booking(
        llm=llm,
        teacher_id=request.teacher_id,
        time_iso=time_iso,
        subject=request.subject,
        lesson_service=lesson_service,
        student_service=student_service,
        user_message=request.message,
    )
    return {
        "tool_called": result["tool_called"],
        "lesson_id": result["lesson_id"],
        "message": result["message"],
    }