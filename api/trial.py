"""
试讲 API（家教场景专属）

按 TUTOR_DESIGN.md §7.1 / §9.1 实现：
- POST /api/trial/book             - 预约试讲
- GET  /api/trial/{id}/feedback    - 获取试讲反馈
- POST /api/trial/{id}/feedback    - 提交试讲反馈
- POST /api/trial/{id}/convert     - 试讲转正式课
- GET  /api/trial/pending          - 待试讲列表
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter(prefix="/api/trial", tags=["试讲管理"])


# ============================================
# Request / Response Models
# ============================================

class TrialBookingRequest(BaseModel):
    teacher_id: int
    student_id: Optional[int] = None
    scheduled_at: datetime
    duration_minutes: int = Field(default=30, ge=15, le=90)
    subject: Optional[str] = None
    student_situation: Optional[str] = None


class TrialBookingResponse(BaseModel):
    success: bool
    lesson_id: Optional[int] = None
    teacher_name: Optional[str] = None
    scheduled_at: Optional[str] = None
    duration_minutes: Optional[int] = None
    message: Optional[str] = None
    conflict_reason: Optional[str] = None


class TrialFeedbackRequest(BaseModel):
    lesson_id: int
    rating: int = Field(..., ge=1, le=5)
    would_continue: bool
    feedback_text: Optional[str] = None
    recommended_subjects: Optional[List[str]] = None


class TrialFeedbackResponse(BaseModel):
    success: bool
    feedback_id: Optional[int] = None
    message: Optional[str] = None


class TrialConversionRequest(BaseModel):
    lesson_id: int
    package_id: int
    weekly_pattern: List[Dict[str, Any]] = Field(
        ..., description='[{"day_of_week": 3, "time": "19:00"}]'
    )
    start_date: str = Field(..., description="YYYY-MM-DD")
    total_weeks: int = Field(default=12, ge=1, le=52)


class TrialConversionResponse(BaseModel):
    success: bool
    series_id: Optional[int] = None
    lessons_created: Optional[int] = None
    skipped: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[str]] = None
    message: Optional[str] = None


# ============================================
# API Endpoints
# ============================================

@router.post("/book", response_model=TrialBookingResponse, summary="预约试讲")
async def book_trial(request: TrialBookingRequest):
    """
    预约一次试讲课
    """
    from services.lesson_service import LessonService
    from services.student_profile_service import StudentProfileService
    from services.teacher_service import TeacherService

    svc = LessonService()
    student_svc = StudentProfileService()
    teacher_svc = TeacherService()

    # 学生（默认 = 当前默认学生）
    if request.student_id:
        student = student_svc.get_student(request.student_id)
    else:
        student = student_svc.initialize_default_student()
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")

    # 老师
    teacher = teacher_svc.get_teacher_by_id(request.teacher_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="老师不存在")
    if not teacher.get("trial_available"):
        raise HTTPException(status_code=400, detail="该老师暂不提供试讲")

    # 时间合理性预检:过去时间直接 400,避免走到 create_lesson 才失败
    if request.scheduled_at < datetime.now():
        raise HTTPException(status_code=400, detail="试讲时间不能早于当前时间")

    # 创建试讲 Lesson (create_lesson 内部已做 teacher/student 冲突检测)
    success, result = svc.create_lesson(
        teacher_id=request.teacher_id,
        student_id=student["id"],
        scheduled_at=request.scheduled_at,
        lesson_type="trial",
        subject=request.subject or (teacher.get("subjects") or [None])[0],
        duration_minutes=request.duration_minutes,
        status="confirmed",
        price=0.0,  # 试讲免费
    )

    if not success:
        # 预约失败也要行为记录(让 'failed_attempt' 也是真实信号)
        _record_behavior_safe(
            action_type="trial_book_failed",
            teacher_id=request.teacher_id,
            action_data={"subject": request.subject, "reason": str(result)},
        )
        return TrialBookingResponse(
            success=False,
            message=f"试讲预约失败：{result}",
            conflict_reason=str(result),
        )

    # 预约成功 -> 记录 trial_book 行为(让行为分析有数据)
    _record_behavior_safe(
        action_type="trial_book",
        teacher_id=request.teacher_id,
        action_data={
            "subject": request.subject,
            "duration_minutes": request.duration_minutes,
            "lesson_id": result,
        },
    )

    return TrialBookingResponse(
        success=True,
        lesson_id=result,
        teacher_name=teacher["name"],
        scheduled_at=request.scheduled_at.isoformat(),
        duration_minutes=request.duration_minutes,
        message=f"已为您预约 {request.scheduled_at.strftime('%Y-%m-%d %H:%M')} "
                f"{teacher['name']}老师的试讲",
    )


@router.post("/{lesson_id}/feedback", response_model=TrialFeedbackResponse, summary="提交试讲反馈")
async def submit_trial_feedback(lesson_id: int, request: TrialFeedbackRequest):
    """
    家长提交试讲后的反馈（评分 + 是否继续 + 文本）
    """
    from agents.trial import TrialFeedbackCollector, ConversionLogic
    from services.student_profile_service import StudentProfileService
    from db.db_router import DatabaseRouter

    collector = TrialFeedbackCollector()
    student_svc = StudentProfileService()
    student = student_svc.initialize_default_student()

    # 拿老师 id 用于行为记录
    db = DatabaseRouter()
    lesson = db.lessons.get_lesson_by_id(lesson_id)
    teacher_id = lesson.get("teacher_id") if lesson else None

    # 提交反馈
    ok, msg, feedback_id = collector.submit_feedback(
        lesson_id=lesson_id,
        parent_id=student["parent_id"],
        rating=request.rating,
        would_continue=request.would_continue,
        feedback_text=request.feedback_text,
        recommended_subjects=request.recommended_subjects,
    )

    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # 反馈成功 -> 记录 trial_feedback 行为
    _record_behavior_safe(
        action_type="trial_feedback",
        teacher_id=teacher_id,
        action_data={
            "rating": request.rating,
            "would_continue": request.would_continue,
            "lesson_id": lesson_id,
        },
    )

    return TrialFeedbackResponse(success=True, feedback_id=feedback_id, message=msg)


@router.get("/{lesson_id}/feedback", summary="获取试讲反馈")
async def get_trial_feedback(lesson_id: int):
    """
    获取某次试讲的反馈内容
    """
    from agents.trial import TrialFeedbackCollector
    collector = TrialFeedbackCollector()
    feedback = collector.get_feedback_summary(lesson_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="该试讲暂无反馈")
    return {"success": True, "data": feedback}


@router.post("/{lesson_id}/convert", response_model=TrialConversionResponse, summary="试讲转正式课")
async def convert_trial_to_series(lesson_id: int, request: TrialConversionRequest):
    """
    试讲满意后，转化为正式系列课：
    1. 创建 LessonSeries
    2. 按 weekly_pattern + total_weeks 生成 N 个 Lesson
    3. 自动跳过冲突时段
    """
    from datetime import date, timedelta
    from agents.lesson import ScheduleGenerator, ConflictDetector
    from db.db_router import DatabaseRouter

    db = DatabaseRouter()
    lesson = db.lessons.get_lesson_by_id(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="试讲课时不存在")
    if lesson.get("lesson_type") != "trial":
        raise HTTPException(status_code=400, detail="该课时不是试讲课")

    package = db.packages.get_package_by_id(request.package_id)
    if not package:
        raise HTTPException(status_code=404, detail="套餐不存在")

    # 老师 availability 用于冲突检查
    teacher = db.teachers.get_teacher_by_id(package["teacher_id"])
    teacher_avail = teacher.get("availability") if teacher else None

    # 生成系列课
    generator = ScheduleGenerator(ConflictDetector())
    start_date = date.fromisoformat(request.start_date)
    normalized_pattern = generator.parse_weekly_pattern(request.weekly_pattern)

    result = generator.generate_series_lessons(
        teacher_id=package["teacher_id"],
        student_id=lesson["student_id"],
        start_date=start_date,
        total_weeks=request.total_weeks,
        weekly_pattern=normalized_pattern,
        duration_minutes=package["duration_minutes"],
        subject=package["subject"],
        package_id=request.package_id,
        teacher_availability=teacher_avail,
        skip_hard_conflicts=True,
    )

    if result.get("error"):
        return TrialConversionResponse(
            success=False,
            message=result["error"],
        )

    # 1. 创建系列记录
    end_date = start_date + timedelta(weeks=request.total_weeks) - timedelta(days=1)
    series_id = db.lessons.add_series(
        teacher_id=package["teacher_id"],
        student_id=lesson["student_id"],
        start_date=start_date,
        end_date=end_date,
        weekly_pattern=normalized_pattern,
        total_lessons=len(result["lessons"]),
        package_id=request.package_id,
    )

    # 2. 批量创建 lesson
    lessons_created = 0
    for lesson_data in result["lessons"]:
        db.lessons.add_lesson(
            teacher_id=lesson_data["teacher_id"],
            student_id=lesson_data["student_id"],
            scheduled_at=lesson_data["scheduled_at"],
            lesson_type="series",
            subject=lesson_data["subject"],
            duration_minutes=lesson_data["duration_minutes"],
            status=lesson_data["status"],
            series_id=series_id,
            series_index=lesson_data["series_index"],
            package_id=lesson_data["package_id"],
        )
        lessons_created += 1

    # 转化成功 -> 记录 trial_convert 行为
    _record_behavior_safe(
        action_type="trial_convert",
        teacher_id=package["teacher_id"],
        action_data={"series_id": series_id, "lessons_created": lessons_created},
    )

    return TrialConversionResponse(
        success=True,
        series_id=series_id,
        lessons_created=lessons_created,
        skipped=result["conflicts"] if result.get("conflicts") else [],
        warnings=result.get("warnings", []),
        message=f"试讲已转化为 {lessons_created} 节系列课",
    )


@router.get("/pending", summary="待试讲列表")
async def list_pending_trials():
    """
    获取所有状态为「已确认、待试讲」的试讲课时
    """
    from services.lesson_service import LessonService
    from services.teacher_service import TeacherService

    svc = LessonService()
    teacher_svc = TeacherService()

    pending = svc.get_pending_trials()
    for item in pending:
        teacher = teacher_svc.get_teacher_by_id(item["teacher_id"])
        item["teacher_name"] = teacher["name"] if teacher else "未知"
        if item.get("scheduled_at") and hasattr(item["scheduled_at"], "isoformat"):
            item["scheduled_at"] = item["scheduled_at"].isoformat()
    return {"success": True, "data": pending, "count": len(pending)}


@router.post("/{lesson_id}/recommend-packages", summary="基于试讲推荐套餐")
async def recommend_packages_after_trial(lesson_id: int):
    """
    试讲完成后，为家长推荐适合孩子 + 老师的套餐
    """
    from agents.trial import ConversionLogic
    from services.student_profile_service import StudentProfileService
    from db.db_router import DatabaseRouter

    db = DatabaseRouter()
    lesson = db.lessons.get_lesson_by_id(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="试讲课时不存在")

    student_svc = StudentProfileService()
    student = student_svc.get_student(lesson["student_id"])

    logic = ConversionLogic()
    packages = logic.recommend_packages(
        student=student,
        teacher_id=lesson["teacher_id"],
        trial_lesson_id=lesson_id,
    )

    return {
        "success": True,
        "data": packages,
        "count": len(packages),
    }


# ============================================
# Helper: 安全地记录行为(失败不影响主业务)
# ============================================

def _record_behavior_safe(
    action_type: str,
    teacher_id: Optional[int] = None,
    action_data: Optional[Dict[str, Any]] = None,
    user_id: str = "default_user",
) -> None:
    """统一封装 UserBehaviorService.record_behavior 调用。
    任何异常都被吞掉, 不影响主业务流; 但记录失败会 log warning."""
    try:
        from services.user_behavior_service import UserBehaviorService
        UserBehaviorService().record_behavior(
            user_id=user_id,
            action_type=action_type,
            action_data=action_data,
            teacher_id=teacher_id,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"行为记录失败({action_type}): {e}")