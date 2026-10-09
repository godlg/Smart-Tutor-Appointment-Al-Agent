"""
退课处理器（含退费计算）

按 TUTOR_DESIGN.md §9 实现：
- 单次课退课：直接取消 Lesson
- 系列课退课：取消整个系列
- 剩余课时退费计算（基于套餐剩余价值）
- 取消原因记录
"""

import logging
import math
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# 退课政策：距离上课时间越近，退费比例越低
REFUND_POLICY = [
    # (提前小时数, 退款比例)
    (48, 1.0),     # 提前 48h 以上：100%
    (24, 0.8),     # 24-48h：80%
    (12, 0.5),     # 12-24h：50%
    (0, 0.0),      # 不到 12h：0%
]


class CancelHandler:
    """退课处理器"""

    def cancel_lesson(
        self,
        lesson_id: int,
        reason: str = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        取消单次课时

        Returns:
            (success, message, refund_info)
        """
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        lesson = db.lessons.get_lesson_by_id(lesson_id)
        if not lesson:
            return False, "课时不存在", None

        if lesson["status"] == "cancelled":
            return False, "课时已是取消状态", None

        if lesson["status"] == "completed":
            return False, "已完成的课时无法取消", None

        # 计算退费
        refund_info = self.calculate_refund(lesson, reason=reason)

        # 更新 Lesson 状态
        update_data = {
            "status": "cancelled",
        }
        if reason:
            update_data["parent_feedback"] = f"[退课原因] {reason}"

        success = db.lessons.update_lesson(lesson_id, **update_data)
        if not success:
            return False, "数据库更新失败", None

        # 如果属于系列，更新系列统计
        if lesson.get("series_id"):
            db.lessons.update_series(
                lesson["series_id"],
                cancelled_lessons=self._get_series_cancelled_count(db, lesson["series_id"]) + 1,
            )

        logger.info(f"退课成功: lesson_id={lesson_id}, refund={refund_info}")
        return True, "退课成功", refund_info

    def cancel_series(
        self,
        series_id: int,
        reason: str = None,
    ) -> Tuple[bool, str]:
        """
        取消整个系列课（取消所有未完成的课时）

        Returns:
            (success, message)
        """
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        lessons = db.lessons.get_lessons_by_series(series_id)
        if not lessons:
            return False, "系列课不存在"

        cancelled = 0
        skipped = 0
        now = datetime.now()
        for lesson in lessons:
            if lesson["status"] in ("cancelled", "completed"):
                skipped += 1
                continue
            db.lessons.update_lesson(lesson["id"], status="cancelled")
            cancelled += 1

        db.lessons.update_series(series_id, status="ended")
        msg = f"系列课已取消：成功取消 {cancelled} 节，跳过 {skipped} 节"
        if reason:
            msg += f"\n原因：{reason}"
        logger.info(msg)
        return True, msg

    def calculate_refund(
        self,
        lesson: Dict[str, Any],
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        计算单次课时退费

        Returns:
            {
                "original_price": float,
                "refund_amount": float,
                "refund_ratio": float,
                "hours_until_lesson": float,
                "policy": str,
            }
        """
        price = lesson.get("price") or 0.0
        if price <= 0:
            return {
                "original_price": 0.0,
                "refund_amount": 0.0,
                "refund_ratio": 1.0,
                "hours_until_lesson": 0.0,
                "policy": "免费课时不涉及退费",
            }

        scheduled_at = lesson.get("scheduled_at")
        if not isinstance(scheduled_at, datetime):
            return {
                "original_price": price,
                "refund_amount": 0.0,
                "refund_ratio": 0.0,
                "hours_until_lesson": 0.0,
                "policy": "时间数据异常，按不退费处理",
            }

        # 用 math.ceil 圆整到小时上沿,避免 (scheduled_at - now) 的 ms 漂移
        # 让"刚好 48h 前"被误判成 [24,48) 档。
        # 例: scheduled_at = now + 48h,真实 (scheduled_at - now) ≈ 47.9999h
        #     ceil 后 = 48,正确落入 [48, ∞) 档;直接用 floor/原始值会落到 [24,48)。
        hours_until = math.ceil(
            (scheduled_at - datetime.now()).total_seconds() / 3600.0
        )

        # 找出适用的退费档
        ratio = 0.0
        policy_desc = ""
        for hours_threshold, refund_ratio in REFUND_POLICY:
            if hours_until >= hours_threshold:
                ratio = refund_ratio
                policy_desc = f"提前 {hours_threshold}h 以上退费 {int(refund_ratio*100)}%"
                break

        # 特殊原因：老师档期问题或系统问题 → 全额退
        if reason and any(kw in reason for kw in ["老师原因", "老师档期", "系统问题", "teacher"]):
            ratio = 1.0
            policy_desc = "老师/系统原因，全额退款"

        refund_amount = round(price * ratio, 2)
        return {
            "original_price": price,
            "refund_amount": refund_amount,
            "refund_ratio": ratio,
            "hours_until_lesson": round(hours_until, 1),
            "policy": policy_desc,
        }

    def _get_series_cancelled_count(self, db, series_id: int) -> int:
        """获取系列已取消课时数"""
        lessons = db.lessons.get_lessons_by_series(series_id)
        return sum(1 for l in lessons if l["status"] == "cancelled")