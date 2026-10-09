"""
流失预警预测器

按 TUTOR_DESIGN.md §5.5 实现：
- 流失风险评分（0-1）
- 基于多维度指标：
  - 缺勤频率
  - 作业完成率（基于 progress）
  - 上次互动距今天数
  - upsell_signal 强度
- 高风险用户列表
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """流失预测器"""

    def __init__(self, student_service=None, progress_service=None):
        self._student_service = student_service
        self._progress_service = progress_service

    @property
    def student_service(self):
        if self._student_service is None:
            from services.student_profile_service import StudentProfileService
            self._student_service = StudentProfileService()
        return self._student_service

    @property
    def progress_service(self):
        if self._progress_service is None:
            from services.progress_service import ProgressService
            self._progress_service = ProgressService()
        return self._progress_service

    # ------------------- 单学生风险评分 -------------------

    def predict_churn_risk(self, student_id: int) -> Dict[str, Any]:
        """
        预测单个学生的流失风险

        Returns:
            {
                "risk_score": 0.0~1.0,
                "risk_level": "low" / "medium" / "high",
                "factors": [{name, value, weight, contribution}, ...],
                "recommendations": [str, ...],
            }
        """
        student = self.student_service.get_student(student_id)
        if not student:
            return {"risk_score": 0.0, "risk_level": "unknown"}

        factors = []
        recommendations = []

        # 因子 1: 上次互动距今天数（权重 0.3）
        days_since_last = self._days_since_last_interaction(student_id)
        days_score = self._score_days_inactive(days_since_last)
        factors.append({
            "name": "上次互动距今",
            "value": f"{days_since_last} 天",
            "raw_score": days_score,
            "weight": 0.30,
        })
        if days_score > 0.5:
            recommendations.append("已较长时间未互动，可主动推送学习反馈")

        # 因子 2: 活跃度（权重 0.25）
        engagement = student.get("engagement_score") or 0.5
        engagement_score = 1.0 - float(engagement)  # 反向
        factors.append({
            "name": "活跃度",
            "value": f"{engagement:.2f}",
            "raw_score": engagement_score,
            "weight": 0.25,
        })
        if engagement_score > 0.5:
            recommendations.append("活跃度偏低，可发送续费或新套餐推荐")

        # 因子 3: 续费信号（权重 0.2）
        upsell_signal = (student.get("upsell_signal") or "none").lower()
        upsell_score = {
            "strong": 0.0,
            "weak": 0.5,
            "none": 0.9,
        }.get(upsell_signal, 0.5)
        factors.append({
            "name": "续费信号",
            "value": upsell_signal,
            "raw_score": upsell_score,
            "weight": 0.20,
        })

        # 因子 4: 作业完成率（基于 ProgressRecord，权重 0.25）
        completion_rate = self._homework_completion_rate(student_id)
        completion_score = 1.0 - completion_rate
        factors.append({
            "name": "作业完成率",
            "value": f"{completion_rate:.1%}",
            "raw_score": completion_score,
            "weight": 0.25,
        })
        if completion_score > 0.4:
            recommendations.append("作业完成率偏低，可联系家长沟通")

        # 加权求和
        total_score = sum(f["raw_score"] * f["weight"] for f in factors)
        total_score = round(min(1.0, max(0.0, total_score)), 3)

        if total_score < 0.3:
            risk_level = "low"
        elif total_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        # 写回数据库
        try:
            self.student_service.update_engagement(student_id, churn_risk=total_score)
        except Exception as e:
            logger.warning(f"更新流失风险失败: {e}")

        return {
            "risk_score": total_score,
            "risk_level": risk_level,
            "factors": factors,
            "recommendations": recommendations,
        }

    # ------------------- 批量分析 -------------------

    def predict_all_students(self) -> List[Dict[str, Any]]:
        """批量分析所有学生的流失风险"""
        all_students = self.student_service.get_all_students()
        results = []
        for s in all_students:
            r = self.predict_churn_risk(s["id"])
            r["student_id"] = s["id"]
            r["student_name"] = s["name"]
            results.append(r)
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results

    def get_high_risk_students(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """获取高风险学生列表"""
        all_results = self.predict_all_students()
        return [r for r in all_results if r["risk_score"] >= threshold]

    # ------------------- 私有方法 -------------------

    def _days_since_last_interaction(self, student_id: int) -> int:
        """获取学生距离上次课时的天数"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        lessons = db.lessons.get_lessons_by_student(student_id)
        if not lessons:
            return 999
        last_lesson = max(lessons, key=lambda l: l["scheduled_at"] or datetime.min)
        if not last_lesson.get("scheduled_at"):
            return 999
        delta = datetime.now() - last_lesson["scheduled_at"]
        return max(0, delta.days)

    def _score_days_inactive(self, days: int) -> float:
        """天数 → 风险分"""
        if days < 7:
            return 0.0
        if days < 14:
            return 0.3
        if days < 30:
            return 0.6
        if days < 60:
            return 0.8
        return 1.0

    def _homework_completion_rate(self, student_id: int) -> float:
        """从 ProgressRecord 中估算作业完成率（简化版）"""
        records = self.progress_service.get_student_progress(student_id, limit=20)
        if not records:
            return 0.5  # 不知道，默认中
        completed = sum(
            1 for r in records
            if (r.get("homework") or "").strip()
        )
        return completed / len(records) if records else 0.5