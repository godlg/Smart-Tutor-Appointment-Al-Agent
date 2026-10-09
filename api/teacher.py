"""
老师 API

提供老师信息查询、筛选、推荐接口
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/teachers", tags=["老师管理"])


class TeacherResponse(BaseModel):
    id: int
    name: str
    gender: Optional[str] = None
    subjects: List[str] = []
    grade_levels: List[str] = []
    teaching_style: Optional[str] = None
    education_bg: Optional[str] = None
    years_experience: int = 0
    hourly_rate: float = 0.0
    rating: float = 5.0
    availability: List[Dict[str, Any]] = []
    trial_available: bool = True
    bio: Optional[str] = None
    highlights: List[str] = []


class RecommendationResponse(BaseModel):
    teacher: TeacherResponse
    score: float
    reason: str


@router.get("/", response_model=List[TeacherResponse], summary="获取所有老师")
async def get_all_teachers(subject: Optional[str] = None, grade_level: Optional[str] = None):
    """获取老师列表，支持按学科/年级筛选"""
    try:
        from services.teacher_service import TeacherService
        svc = TeacherService()
        svc.initialize_default_teachers()

        if subject and grade_level:
            teachers = svc.find_by_subject_and_grade(subject, grade_level)
        elif subject:
            teachers = svc.find_by_subject(subject)
        else:
            teachers = svc.get_all_teachers()

        return [TeacherResponse(**t) for t in teachers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取老师失败: {str(e)}")


@router.get("/subjects", summary="获取所有学科")
async def get_all_subjects():
    from services.teacher_service import TeacherService
    svc = TeacherService()
    return {"subjects": svc.get_all_subjects()}


@router.get("/grade-levels", summary="获取所有年级段")
async def get_all_grade_levels():
    from services.teacher_service import TeacherService
    svc = TeacherService()
    return {"grade_levels": svc.get_all_grade_levels()}


@router.get("/{teacher_id}", response_model=TeacherResponse, summary="获取单个老师")
async def get_teacher(teacher_id: int):
    try:
        from services.teacher_service import TeacherService
        svc = TeacherService()
        teacher = svc.get_teacher_by_id(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail="老师不存在")
        return TeacherResponse(**teacher)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{teacher_id}/availability", summary="获取老师可用时段")
async def get_teacher_availability(teacher_id: int):
    try:
        from services.teacher_service import TeacherService
        svc = TeacherService()
        teacher = svc.get_teacher_by_id(teacher_id)
        if not teacher:
            raise HTTPException(status_code=404, detail="老师不存在")
        return {"teacher_id": teacher_id, "availability": teacher.get("availability", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=List[RecommendationResponse], summary="智能推荐老师")
async def recommend_teachers(request: Dict[str, Any]):
    """根据学生画像智能推荐老师"""
    try:
        from services.recommendation_service import RecommendationService
        from services.student_profile_service import StudentProfileService

        subject = request.get("subject")
        grade_level = request.get("grade_level")
        student_id = request.get("student_id")
        top_k = request.get("top_k", 3)

        student_svc = StudentProfileService()
        if student_id:
            student = student_svc.get_student(student_id)
        else:
            student = student_svc.initialize_default_student()

        if grade_level and not student.get("grade_level"):
            student["grade_level"] = grade_level

        rec_svc = RecommendationService()
        recommendations = rec_svc.recommend_teachers(student, subject=subject, top_k=top_k)

        return [
            RecommendationResponse(
                teacher=TeacherResponse(**r["teacher"]),
                score=r["score"],
                reason=r["reason"],
            )
            for r in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")