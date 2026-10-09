"""
套餐 API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/packages", tags=["套餐管理"])


class PackageResponse(BaseModel):
    id: int
    name: str
    teacher_id: int
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    total_lessons: int
    duration_minutes: int
    price_total: float
    price_per_lesson: float
    weekly_schedule: List[Dict[str, Any]] = []
    validity_days: int
    description: Optional[str] = None
    is_active: bool


@router.get("/", response_model=List[PackageResponse], summary="获取套餐列表")
async def get_packages(
    subject: Optional[str] = None,
    grade_level: Optional[str] = None,
    teacher_id: Optional[int] = None,
):
    try:
        from services.package_service import PackageService
        svc = PackageService()
        svc.initialize_default_packages()
        packages = svc.get_all_packages(subject=subject, grade_level=grade_level, teacher_id=teacher_id)
        return [PackageResponse(**p) for p in packages]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{package_id}", response_model=PackageResponse, summary="获取单个套餐")
async def get_package(package_id: int):
    try:
        from services.package_service import PackageService
        svc = PackageService()
        p = svc.get_package(package_id)
        if not p:
            raise HTTPException(status_code=404, detail="套餐不存在")
        return PackageResponse(**p)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{package_id}/purchase", summary="购买套餐（Mock）")
async def purchase_package(package_id: int, student_id: int = 1):
    """MVP 阶段 Mock，直接返回 success"""
    try:
        from services.package_service import PackageService
        from services.student_profile_service import StudentProfileService

        svc = PackageService()
        package = svc.get_package(package_id)
        if not package:
            raise HTTPException(status_code=404, detail="套餐不存在")

        student_svc = StudentProfileService()
        student = student_svc.get_student(student_id)
        if not student:
            student = student_svc.initialize_default_student()

        # MVP: 直接返回购买成功，不做支付
        return {
            "success": True,
            "package_id": package_id,
            "student_id": student["id"],
            "amount": package["price_total"],
            "message": "购买成功（Mock）",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))