"""
续费信号检测器

按 TUTOR_DESIGN.md §5.5 / §9.4 实现：
- 触发条件：套餐剩余课时 ≤ 3 或 距离过期 < 14 天
- 检测学生 upsell_signal 强度
- 生成个性化续费推荐语
- 推荐适合的续费套餐
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# 续费信号阈值
LOW_LESSONS_THRESHOLD = 3           # 剩余课时低于该值触发续费提醒
NEAR_EXPIRY_DAYS = 14               # 距离过期少于该天数触发
ENGAGEMENT_ACTIVE = 0.6             # 活跃度阈值


class RenewalDetector:
    """续费提醒检测器"""

    def __init__(self, package_service=None, student_service=None, teacher_service=None):
        self._package_service = package_service
        self._student_service = student_service
        self._teacher_service = teacher_service

    @property
    def package_service(self):
        if self._package_service is None:
            from services.package_service import PackageService
            self._package_service = PackageService()
        return self._package_service

    @property
    def student_service(self):
        if self._student_service is None:
            from services.student_profile_service import StudentProfileService
            self._student_service = StudentProfileService()
        return self._student_service

    @property
    def teacher_service(self):
        if self._teacher_service is None:
            from services.teacher_service import TeacherService
            self._teacher_service = TeacherService()
        return self._teacher_service

    # ------------------- 触发检测 -------------------

    def should_trigger_renewal(self, student_id: int) -> Dict[str, Any]:
        """
        判断是否需要触发续费提醒

        Returns:
            {
                "should_trigger": bool,
                "reason": str,
                "remaining_lessons": int,
                "days_until_expiry": int,
                "urgency": "high" / "medium" / "low",
            }
        """
        # 简化实现：通过 lesson_service 统计学生课时
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()

        # 找到学生最近的 series
        series_list = db.lessons.get_series_by_student(student_id)
        if not series_list:
            return {"should_trigger": False, "reason": "暂无进行中的套餐"}

        latest_series = max(series_list, key=lambda s: s.get("created_at") or date.min)
        total = latest_series.get("total_lessons", 0) or 0
        completed = latest_series.get("completed_lessons", 0) or 0
        cancelled = latest_series.get("cancelled_lessons", 0) or 0
        remaining = max(0, total - completed - cancelled)

        # 距离过期
        end_date = latest_series.get("end_date")
        days_until_expiry = None
        if isinstance(end_date, date):
            days_until_expiry = (end_date - date.today()).days

        should = False
        reasons = []
        urgency = "low"

        if remaining <= LOW_LESSONS_THRESHOLD:
            should = True
            reasons.append(f"套餐仅剩 {remaining} 课时")
            urgency = "high" if remaining <= 1 else "medium"

        if days_until_expiry is not None and days_until_expiry <= NEAR_EXPIRY_DAYS:
            should = True
            reasons.append(f"距离套餐过期还有 {days_until_expiry} 天")
            if days_until_expiry <= 3:
                urgency = "high"

        return {
            "should_trigger": should,
            "reason": "；".join(reasons) if reasons else "暂不需要续费",
            "remaining_lessons": remaining,
            "days_until_expiry": days_until_expiry,
            "urgency": urgency,
            "latest_series_id": latest_series.get("id"),
            "package_id": latest_series.get("package_id"),
        }

    # ------------------- 续费推荐 -------------------

    def recommend_renewal_packages(
        self,
        student_id: int,
        current_package_id: int = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        推荐续费套餐

        策略：
        - 默认推荐同老师的更长期套餐（升单）
        - 考虑学生最近进步情况（基于 progress）
        """
        student = self.student_service.get_student(student_id)
        if not student:
            return []

        teacher_id = None
        if current_package_id:
            pkg = self.package_service.get_package(current_package_id)
            if pkg:
                teacher_id = pkg["teacher_id"]

        # 优先推同老师套餐
        if teacher_id:
            candidates = self.package_service.get_all_packages(teacher_id=teacher_id)
        else:
            candidates = self.package_service.get_all_packages(
                subject=(student.get("weak_subjects") or [None])[0],
                grade_level=student.get("grade_level"),
            )

        # 过滤掉当前已购套餐
        if current_package_id:
            candidates = [c for c in candidates if c["id"] != current_package_id]

        if not candidates:
            return []

        # 简化排序：总课时数升序（推荐「继续」而不是「重买」）
        candidates.sort(key=lambda p: (p.get("total_lessons") or 0, p.get("price_total") or 0))
        return candidates[:limit]

    # ------------------- 消息生成 -------------------

    def generate_renewal_message(
        self,
        student_id: int,
        trigger_info: Dict[str, Any] = None,
    ) -> str:
        """
        生成个性化续费提醒消息
        """
        if trigger_info is None:
            trigger_info = self.should_trigger_renewal(student_id)

        if not trigger_info.get("should_trigger"):
            return ""

        student = self.student_service.get_student(student_id)
        student_name = student.get("name", "孩子") if student else "孩子"

        # 推荐续费套餐
        renewal_packages = self.recommend_renewal_packages(
            student_id=student_id,
            current_package_id=trigger_info.get("package_id"),
        )

        urgency_emoji = {
            "high": "🚨",
            "medium": "⏰",
            "low": "💡",
        }.get(trigger_info.get("urgency"), "💡")

        lines = [
            f"\n顾问：{urgency_emoji} 续费提醒\n",
            f"{student_name}的辅导套餐：{trigger_info['reason']}",
            f"\n为了保持学习连贯性，建议您提前续费。",
        ]

        if renewal_packages:
            lines.append(f"\n推荐续费套餐：")
            for idx, pkg in enumerate(renewal_packages[:3], 1):
                lines.append(
                    f"{idx}. {pkg['name']} - "
                    f"¥{pkg['price_total']}（{pkg['total_lessons']}课时）"
                )
            lines.append(f"\n回复「续费 {renewal_packages[0]['name']}」即可购买。")
        else:
            lines.append(
                f"\n您可以在「/packages」页面查看所有套餐，或告诉我孩子的需求，"
                f"我为您推荐合适的。"
            )

        return "\n".join(lines)