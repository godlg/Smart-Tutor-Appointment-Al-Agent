"""
学生 + 家长 API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/students", tags=["学生管理"])


class StudentResponse(BaseModel):
    id: int
    parent_id: int
    name: str
    grade_level: Optional[str] = None
    weak_subjects: List[str] = []
    learning_goals: List[str] = []
    preferred_style: Optional[str] = None
    school: Optional[str] = None
    notes: Optional[str] = None


class CreateStudentRequest(BaseModel):
    name: str
    grade_level: str
    weak_subjects: List[str] = []
    learning_goals: List[str] = []
    preferred_style: Optional[str] = None
    school: Optional[str] = None


@router.get("/", response_model=List[StudentResponse], summary="获取所有学生")
async def get_all_students():
    try:
        from services.student_profile_service import StudentProfileService
        svc = StudentProfileService()
        students = svc.get_all_students()
        return [StudentResponse(**s) for s in students]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default", response_model=StudentResponse, summary="获取默认学生")
async def get_default_student():
    try:
        from services.student_profile_service import StudentProfileService
        svc = StudentProfileService()
        student = svc.initialize_default_student()
        return StudentResponse(**student)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{student_id}", response_model=StudentResponse, summary="获取单个学生")
async def get_student(student_id: int):
    try:
        from services.student_profile_service import StudentProfileService
        svc = StudentProfileService()
        student = svc.get_student(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="学生不存在")
        return StudentResponse(**student)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", summary="创建学生")
async def create_student(request: CreateStudentRequest):
    try:
        from services.student_profile_service import StudentProfileService
        svc = StudentProfileService()
        parent = svc.get_or_create_default_parent()
        student_id = svc.add_student(
            parent_id=parent["id"],
            name=request.name,
            grade_level=request.grade_level,
            weak_subjects=request.weak_subjects,
            learning_goals=request.learning_goals,
            preferred_style=request.preferred_style,
            school=request.school,
        )
        return {"success": True, "student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))