"""老师面 API(老师面最小闭环)

- GET  /api/teacher/me                       当前老师身份
- GET  /api/teacher/lessons?status=pending   本老师的课(只含本老师)
- POST /api/teacher/lessons/{id}/accept      确认档期 pending → confirmed
- POST /api/teacher/lessons/{id}/reject      拒绝 pending → cancelled(不走退费)

红线:teacher_id 只来自会话(require_teacher),绝不接受请求体 / query / 路径里的 teacher_id。
归属不符一律 404(不区分「不存在」与「不是你的」,避免探测他人资源)。

注意:本文件不是 api/teacher.py(那是公开的 /api/teachers 老师列表,已存在)。
"""

import logging
from datetime import timedelta
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.lesson_service import LessonService

from .core.console_auth import require_teacher

logger = logging.getLogger(__name__)

_require_teacher = require_teacher()
router = APIRouter(
    prefix="/api/teacher",
    tags=["老师面"],
    dependencies=[Depends(_require_teacher)],
)

LessonStatusQuery = Literal["pending", "confirmed", "completed", "cancelled"]


class RejectRequest(BaseModel):
    """拒绝请求体;多余字段(包括试图传入的 teacher_id)被 pydantic 忽略。"""
    reason: Optional[str] = None


def get_lesson_service() -> LessonService:
    """课时服务工厂;测试可通过 monkeypatch services.lesson_service.DatabaseRouter 指向临时库。"""
    return LessonService()


def _owned_lesson(service: LessonService, lesson_id: int, teacher_id: int) -> Dict[str, Any]:
    """取课并校验归属;不存在或不属于本老师一律 404。"""
    lesson = service.get_lesson(lesson_id)
    if lesson is None or lesson["teacher_id"] != teacher_id:
        raise HTTPException(status_code=404, detail="课时不存在")
    return lesson


def _ensure_pending(lesson: Dict[str, Any]) -> None:
    if lesson["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"课时状态为 {lesson['status']}，仅待确认课时可操作",
        )


def _student_name(service: LessonService, student_id: Optional[int]) -> Optional[str]:
    if student_id is None:
        return None
    student = service.db.students.get_student_by_id(student_id)
    return student["name"] if student else None


@router.get("/me", summary="当前老师身份")
def get_me(
    teacher: dict = Depends(_require_teacher),
    service: LessonService = Depends(get_lesson_service),
):
    teacher_row = service.teacher_repo.get_teacher_by_id(teacher["teacher_id"])
    if teacher_row is None:
        raise HTTPException(status_code=404, detail="老师档案不存在")
    return {
        "teacher_id": teacher["teacher_id"],
        "name": teacher_row["name"],
        "username": teacher["username"],
    }


@router.get("/lessons", summary="本老师的课时列表")
def list_my_lessons(
    status: LessonStatusQuery = "pending",
    teacher: dict = Depends(_require_teacher),
    service: LessonService = Depends(get_lesson_service),
):
    lessons = service.get_lessons_by_teacher(teacher["teacher_id"], status=status)
    return [
        {
            "id": lesson["id"],
            "scheduled_at": lesson["scheduled_at"],
            "subject": lesson["subject"],
            "lesson_type": lesson["lesson_type"],
            "student_id": lesson["student_id"],
            "student_name": _student_name(service, lesson["student_id"]),
            "duration_minutes": lesson["duration_minutes"],
            "status": lesson["status"],
        }
        for lesson in lessons
    ]


@router.post("/lessons/{lesson_id}/accept", summary="确认档期")
def accept_lesson(
    lesson_id: int,
    teacher: dict = Depends(_require_teacher),
    service: LessonService = Depends(get_lesson_service),
):
    lesson = _owned_lesson(service, lesson_id, teacher["teacher_id"])
    _ensure_pending(lesson)

    # 确认前复检档期:老师可能在列表加载后已有其他确认课
    end_time = lesson["scheduled_at"] + timedelta(minutes=lesson["duration_minutes"])
    if service.lesson_repo.check_teacher_conflict(
        lesson["teacher_id"], lesson["scheduled_at"], end_time, exclude_lesson_id=lesson_id
    ):
        raise HTTPException(status_code=409, detail="该时段与你已有课程冲突，无法确认")

    service.update_lesson_status(lesson_id, "confirmed")
    return {"lesson_id": lesson_id, "status": "confirmed"}


@router.post("/lessons/{lesson_id}/reject", summary="拒绝档期")
def reject_lesson(
    lesson_id: int,
    body: RejectRequest,
    teacher: dict = Depends(_require_teacher),
    service: LessonService = Depends(get_lesson_service),
):
    lesson = _owned_lesson(service, lesson_id, teacher["teacher_id"])
    _ensure_pending(lesson)

    # 只做状态变更。禁止调用 cancel_lesson():它走退费语义(见 agents/lesson/cancel_handler.py)
    service.update_lesson_status(lesson_id, "cancelled")
    logger.info(
        "老师拒绝待确认课时",
        extra={
            "event": "teacher_lesson_rejected",
            "teacher_id": teacher["teacher_id"],
            "lesson_id": lesson_id,
            "reason": body.reason,
        },
    )
    return {"lesson_id": lesson_id, "status": "cancelled", "reason": body.reason}
