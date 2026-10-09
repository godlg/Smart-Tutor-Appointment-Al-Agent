"""
学习进度 API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/progress", tags=["进度管理"])


class ProgressResponse(BaseModel):
    id: int
    lesson_id: int
    student_id: int
    teacher_id: int
    content_covered: Optional[str] = None
    mastery_score: Optional[float] = None
    homework: Optional[str] = None
    weak_points: List[str] = []
    ai_summary: Optional[str] = None
    teacher_comment: Optional[str] = None
    parent_comment: Optional[str] = None
    created_at: Optional[str] = None


class CreateProgressRequest(BaseModel):
    lesson_id: int
    student_id: int
    teacher_id: int
    content_covered: str
    mastery_score: float
    homework: Optional[str] = None
    weak_points: List[str] = []
    teacher_comment: Optional[str] = None


@router.get("/student/{student_id}", response_model=List[ProgressResponse], summary="获取学生的进度记录")
async def get_student_progress(student_id: int, limit: int = 20):
    try:
        from services.progress_service import ProgressService
        svc = ProgressService()
        records = svc.get_student_progress(student_id, limit)
        return [_to_response(r) for r in records]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", summary="创建进度记录")
async def create_progress(request: CreateProgressRequest):
    try:
        from services.progress_service import ProgressService
        svc = ProgressService()
        progress_id = svc.add_progress(**request.dict())
        return {"success": True, "progress_id": progress_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _to_response(r: Dict[str, Any]) -> ProgressResponse:
    created = r.get("created_at")
    if created and hasattr(created, "isoformat"):
        created = created.isoformat()
    return ProgressResponse(
        id=r["id"],
        lesson_id=r["lesson_id"],
        student_id=r["student_id"],
        teacher_id=r["teacher_id"],
        content_covered=r.get("content_covered"),
        mastery_score=r.get("mastery_score"),
        homework=r.get("homework"),
        weak_points=r.get("weak_points", []),
        ai_summary=r.get("ai_summary"),
        teacher_comment=r.get("teacher_comment"),
        parent_comment=r.get("parent_comment"),
        created_at=created,
    )