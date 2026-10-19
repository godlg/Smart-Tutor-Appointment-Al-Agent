"""
课时 API（家教场景）

按 TUTOR_DESIGN.md §7.1 实现：
- 单次课 CRUD
- 系列课生成（POST /api/lessons/series）
- 调课（POST /api/lessons/{id}/reschedule，含级联）
- 退课（POST /api/lessons/{id}/cancel，含退费）
- 系列课查询（GET /api/lesson-series/{id}）
- 冲突预览（POST /api/lessons/{id}/reschedule-preview）
- 调课请求（POST/GET /api/lessons/{id}/change-requests，C2a：待老师确认，不立即改期）
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Literal, Optional
from datetime import datetime, date as DateType
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/lessons", tags=["课时管理"])
# 系列课独立前缀（与设计文档一致）
series_router = APIRouter(prefix="/api/lesson-series", tags=["系列课管理"])


class LessonResponse(BaseModel):
    id: int
    teacher_id: int
    student_id: int
    scheduled_at: str
    duration_minutes: int
    lesson_type: str
    subject: Optional[str] = None
    status: str
    series_id: Optional[int] = None
    price: Optional[float] = None


class CreateLessonRequest(BaseModel):
    teacher_id: int
    student_id: int
    scheduled_at: datetime
    duration_minutes: int = 60
    subject: Optional[str] = None
    lesson_type: str = "regular"
    price: Optional[float] = None


class CreateSeriesRequest(BaseModel):
    teacher_id: int
    student_id: int
    weekly_pattern: List[Dict[str, Any]] = Field(..., description='[{"day_of_week": 3, "time": "19:00"}]')
    start_date: str = Field(..., description="YYYY-MM-DD")
    total_weeks: int
    subject: Optional[str] = None
    duration_minutes: int = 60
    package_id: Optional[int] = None


class RescheduleRequest(BaseModel):
    new_time: datetime
    mode: Optional[str] = Field(
        default="only_this",
        description="only_this | this_and_following",
    )


class CancelRequest(BaseModel):
    reason: Optional[str] = None


class SeriesPreviewRequest(BaseModel):
    teacher_id: int
    student_id: int
    weekly_pattern: List[Dict[str, Any]] = Field(..., description='[{"day_of_week": 3, "time": "19:00"}]')
    start_date: str = Field(..., description="YYYY-MM-DD")
    total_weeks: int
    subject: Optional[str] = None
    duration_minutes: int = 60
    package_id: Optional[int] = None


@router.get("/", response_model=List[LessonResponse], summary="获取课时列表")
async def get_lessons(
    teacher_id: Optional[int] = None,
    student_id: Optional[int] = None,
    status: Optional[str] = None,
):
    try:
        from services.lesson_service import LessonService
        svc = LessonService()
        if teacher_id:
            lessons = svc.get_lessons_by_teacher(teacher_id, status=status)
        elif student_id:
            lessons = svc.get_lessons_by_student(student_id, status=status)
        else:
            from services.student_profile_service import StudentProfileService
            student = StudentProfileService().initialize_default_student()
            lessons = svc.get_lessons_by_student(student["id"])
        return [_lesson_to_response(l) for l in lessons]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{lesson_id}", response_model=LessonResponse, summary="获取单个课时")
async def get_lesson(lesson_id: int):
    try:
        from services.lesson_service import LessonService
        svc = LessonService()
        lesson = svc.get_lesson(lesson_id)
        if not lesson:
            raise HTTPException(status_code=404, detail="课时不存在")
        return _lesson_to_response(lesson)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", summary="创建课时")
async def create_lesson(request: CreateLessonRequest):
    try:
        from services.lesson_service import LessonService
        svc = LessonService()
        success, result = svc.create_lesson(
            teacher_id=request.teacher_id,
            student_id=request.student_id,
            scheduled_at=request.scheduled_at,
            duration_minutes=request.duration_minutes,
            subject=request.subject,
            lesson_type=request.lesson_type,
            price=request.price,
        )
        if success:
            return {"success": True, "lesson_id": result}
        return {"success": False, "error": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{lesson_id}/reschedule", summary="调课")
async def reschedule_lesson(lesson_id: int, request: RescheduleRequest):
    """
    调课，支持单次和系列级联
    mode:
      - only_this: 只调这一次
      - this_and_following: 调这次及之后所有周（系列级联）
    """
    try:
        from agents.lesson import RescheduleHandler, RescheduleMode

        mode = RescheduleMode.ONLY_THIS
        if request.mode == "this_and_following":
            mode = RescheduleMode.THIS_AND_FOLLOWING

        handler = RescheduleHandler()
        success, message = handler.reschedule(
            lesson_id=lesson_id,
            new_time=request.new_time,
            mode=mode,
        )
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{lesson_id}/reschedule-preview", summary="调课预览（不实际修改）")
async def reschedule_lesson_preview(lesson_id: int, request: RescheduleRequest):
    """
    预览调课结果（特别是级联调课的影响）
    """
    try:
        from agents.lesson import RescheduleHandler

        handler = RescheduleHandler()
        result = handler.preview_cascade(
            lesson_id=lesson_id,
            new_time=request.new_time,
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{lesson_id}/cancel", summary="取消课时（含退费计算）")
async def cancel_lesson(lesson_id: int, request: CancelRequest = None):
    """
    取消课时并计算退费
    """
    try:
        from agents.lesson import CancelHandler

        handler = CancelHandler()
        reason = request.reason if request else None
        success, message, refund_info = handler.cancel_lesson(
            lesson_id=lesson_id,
            reason=reason,
        )
        return {
            "success": success,
            "message": message,
            "refund": refund_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/preview", summary="预览系列课生成（不实际创建）")
async def preview_series(request: SeriesPreviewRequest):
    """
    预览系列课生成结果，包括冲突和警告
    """
    try:
        from agents.lesson import ScheduleGenerator, ConflictDetector
        from db.db_router import DatabaseRouter

        db = DatabaseRouter()
        teacher_avail = None
        teacher = db.teachers.get_teacher_by_id(request.teacher_id)
        if teacher:
            teacher_avail = teacher.get("availability")

        generator = ScheduleGenerator(ConflictDetector())
        normalized_pattern = generator.parse_weekly_pattern(request.weekly_pattern)
        start_date = DateType.fromisoformat(request.start_date)

        result = generator.generate_series_lessons(
            teacher_id=request.teacher_id,
            student_id=request.student_id,
            start_date=start_date,
            total_weeks=request.total_weeks,
            weekly_pattern=normalized_pattern,
            duration_minutes=request.duration_minutes,
            subject=request.subject,
            package_id=request.package_id,
            teacher_availability=teacher_avail,
            skip_hard_conflicts=True,
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/{series_id}/cancel", summary="取消整个系列课")
async def cancel_series(series_id: int, request: CancelRequest = None):
    """
    取消整个系列课（所有未完成课时）
    """
    try:
        from agents.lesson import CancelHandler

        handler = CancelHandler()
        reason = request.reason if request else None
        success, message = handler.cancel_series(series_id=series_id, reason=reason)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series", summary="创建系列课（周固定）")
async def create_series(request: CreateSeriesRequest):
    try:
        from datetime import date
        from services.lesson_service import LessonService
        svc = LessonService()
        start_date = date.fromisoformat(request.start_date)
        success, result = svc.create_lesson_series(
            teacher_id=request.teacher_id,
            student_id=request.student_id,
            weekly_pattern=request.weekly_pattern,
            start_date=start_date,
            total_weeks=request.total_weeks,
            subject=request.subject,
            duration_minutes=request.duration_minutes,
            package_id=request.package_id,
        )
        return {"success": success, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/trials", summary="获取待试讲课时（保留兼容）")
async def get_pending_trials():
    """
    已迁移到 api/trial.py 的 /api/trial/pending，
    此接口保留供旧调用方使用。
    """
    try:
        from services.lesson_service import LessonService
        svc = LessonService()
        return svc.get_pending_trials()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _lesson_to_response(lesson: Dict[str, Any]) -> LessonResponse:
    scheduled = lesson.get("scheduled_at")
    if isinstance(scheduled, datetime):
        scheduled = scheduled.isoformat()
    return LessonResponse(
        id=lesson["id"],
        teacher_id=lesson["teacher_id"],
        student_id=lesson["student_id"],
        scheduled_at=scheduled,
        duration_minutes=lesson["duration_minutes"],
        lesson_type=lesson["lesson_type"],
        subject=lesson.get("subject"),
        status=lesson["status"],
        series_id=lesson.get("series_id"),
        price=lesson.get("price"),
    )


# ============================================
# 系列课独立路由（/api/lesson-series/{id}）
# ============================================

class SeriesDetailResponse(BaseModel):
    id: int
    teacher_id: int
    student_id: int
    package_id: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    weekly_pattern: List[Dict[str, Any]] = []
    total_lessons: int = 0
    completed_lessons: int = 0
    cancelled_lessons: int = 0
    status: str = "active"
    lessons: List[LessonResponse] = []


@series_router.get("/{series_id}", response_model=SeriesDetailResponse, summary="获取系列课详情")
async def get_series_detail(series_id: int):
    """
    获取一个 LessonSeries 的完整信息：
    - 系列元数据
    - 所有 lesson 列表
    """
    try:
        from db.db_router import DatabaseRouter
        from services.teacher_service import TeacherService

        db = DatabaseRouter()
        series = db.lessons.get_series_by_id(series_id)
        if not series:
            raise HTTPException(status_code=404, detail="系列课不存在")

        lessons = db.lessons.get_lessons_by_series(series_id)

        # 格式化日期
        start_date = series.get("start_date")
        end_date = series.get("end_date")
        if hasattr(start_date, "isoformat"):
            start_date = start_date.isoformat()
        if hasattr(end_date, "isoformat"):
            end_date = end_date.isoformat()

        # 格式化课时
        lesson_responses = [_lesson_to_response(l) for l in lessons]

        return SeriesDetailResponse(
            id=series["id"],
            teacher_id=series["teacher_id"],
            student_id=series["student_id"],
            package_id=series.get("package_id"),
            start_date=start_date,
            end_date=end_date,
            weekly_pattern=series.get("weekly_pattern", []),
            total_lessons=series.get("total_lessons", 0),
            completed_lessons=series.get("completed_lessons", 0),
            cancelled_lessons=series.get("cancelled_lessons", 0),
            status=series.get("status", "active"),
            lessons=lesson_responses,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@series_router.get("/", summary="获取学生的所有系列课")
async def list_student_series(student_id: Optional[int] = None):
    """
    获取学生（或默认学生）的所有系列课
    """
    try:
        from db.db_router import DatabaseRouter
        from services.student_profile_service import StudentProfileService

        db = DatabaseRouter()
        student_svc = StudentProfileService()

        if student_id is None:
            student = student_svc.initialize_default_student()
            student_id = student["id"]

        series_list = db.lessons.get_series_by_student(student_id)
        for s in series_list:
            if hasattr(s.get("start_date"), "isoformat"):
                s["start_date"] = s["start_date"].isoformat()
            if hasattr(s.get("end_date"), "isoformat"):
                s["end_date"] = s["end_date"].isoformat()

        return {"success": True, "data": series_list, "count": len(series_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 调课请求（C2a：调课需老师确认）
# ============================================

class ChangeRequestCreate(BaseModel):
    new_time: datetime
    mode: Literal["only_this", "this_and_following"] = "only_this"
    # 未鉴权:与本文件其它 C 端端点一致,调用方可自报身份
    requested_by: Optional[str] = "parent"


@router.post("/{lesson_id}/change-requests", summary="提交调课请求（待老师确认，不立即改期）")
async def create_change_request(lesson_id: int, request: ChangeRequestCreate):
    from services.lesson_change_service import LessonChangeError, LessonChangeService

    service = LessonChangeService()
    try:
        result = service.create_request(
            lesson_id=lesson_id,
            new_time=request.new_time,
            mode=request.mode,
            requested_by=request.requested_by or "parent",
        )
        return {"success": True, **result}
    except LessonChangeError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    finally:
        service.close()


@router.get("/{lesson_id}/change-requests", summary="课时的调课请求列表（按创建时间倒序）")
async def list_change_requests(lesson_id: int):
    from services.lesson_change_service import LessonChangeService

    service = LessonChangeService()
    try:
        return {"success": True, "data": service.list_for_lesson(lesson_id)}
    finally:
        service.close()