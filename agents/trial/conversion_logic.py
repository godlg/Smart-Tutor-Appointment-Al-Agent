"""
试讲转化策略

按 TUTOR_DESIGN.md §5.3 / §9.1 实现：
- 试讲反馈收集
- 流失原因分类
- 转化为正式课的套餐推荐
- 流失后画像更新
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ConversionStrategy(str):
    """转化策略"""
    RECOMMEND_PACKAGE = "recommend_package"   # 推荐套餐
    REMIND_LATER = "remind_later"              # 稍后提醒
    CHURN = "churn"                            # 流失


class ConversionOutcome(str):
    """转化结果"""
    CONVERTED = "converted"           # 成功转化
    PENDING = "pending"               # 待定
    CHURNED = "churned"               # 流失
    REMIND_LATER = "remind_later"     # 延期


class ConversionLogic:
    """试讲转化处理器"""

    def __init__(self, package_service=None, teacher_service=None, student_service=None):
        self._package_service = package_service
        self._teacher_service = teacher_service
        self._student_service = student_service

    @property
    def package_service(self):
        if self._package_service is None:
            from services.package_service import PackageService
            self._package_service = PackageService()
        return self._package_service

    @property
    def teacher_service(self):
        if self._teacher_service is None:
            from services.teacher_service import TeacherService
            self._teacher_service = TeacherService()
        return self._teacher_service

    @property
    def student_service(self):
        if self._student_service is None:
            from services.student_profile_service import StudentProfileService
            self._student_service = StudentProfileService()
        return self._student_service

    # ------------------- 推荐套餐 -------------------

    def recommend_packages(
        self,
        student: Dict[str, Any],
        teacher_id: int,
        trial_lesson_id: int = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        基于学生画像 + 试讲老师，推荐适合的套餐

        排序规则：
        1. 学科匹配
        2. 年级匹配
        3. 价格适配（不要超过家长预算的 1.5 倍）
        4. 课时数适配（参考学生薄弱学科数量）
        """
        all_packages = self.package_service.get_all_packages(teacher_id=teacher_id)
        if not all_packages:
            # 找不到该老师的套餐，扩大范围：按学科+年级
            weak_subs = student.get("weak_subjects") or []
            grade_level = student.get("grade_level")
            candidates = []
            for sub in weak_subs:
                candidates.extend(
                    self.package_service.get_all_packages(subject=sub, grade_level=grade_level)
                )
            # 去重
            seen = set()
            all_packages = []
            for p in candidates:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    all_packages.append(p)

        if not all_packages:
            return []

        scored = []
        for pkg in all_packages:
            score = self._score_package(pkg, student)
            scored.append({"package": pkg, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [
            {
                **item["package"],
                "match_score": round(item["score"], 1),
                "match_reason": self._build_match_reason(item["package"], student),
            }
            for item in scored[:limit]
        ]

    def _score_package(self, pkg: Dict[str, Any], student: Dict[str, Any]) -> float:
        score = 0.0
        weak_subjects = student.get("weak_subjects") or []
        grade_level = student.get("grade_level")

        # 学科匹配
        if pkg.get("subject") and pkg["subject"] in weak_subjects:
            score += 40
        elif pkg.get("subject"):
            score += 10

        # 年级匹配
        if grade_level and pkg.get("grade_level") == grade_level:
            score += 30

        # 价格适配
        parent_budget = (student.get("budget_max") or 500.0)
        if pkg.get("price_total"):
            if pkg["price_total"] <= parent_budget * 4:
                score += 20
            elif pkg["price_total"] <= parent_budget * 6:
                score += 10

        # 课时数：薄弱学科越多，越推荐多课时套餐
        weak_count = len(weak_subjects)
        if pkg.get("total_lessons", 0) >= 16 and weak_count >= 2:
            score += 15

        return score

    def _build_match_reason(self, pkg: Dict[str, Any], student: Dict[str, Any]) -> str:
        reasons = []
        if pkg.get("subject") in (student.get("weak_subjects") or []):
            reasons.append(f"针对孩子薄弱学科 {pkg['subject']}")
        if pkg.get("grade_level") == student.get("grade_level"):
            reasons.append(f"匹配 {pkg['grade_level']} 年级")
        if pkg.get("total_lessons", 0) >= 24:
            reasons.append("课时充足，长期规划")
        elif pkg.get("total_lessons", 0) >= 12:
            reasons.append("标准套餐")
        return "；".join(reasons) if reasons else "综合推荐"

    # ------------------- 流失原因分析 -------------------

    def classify_churn_reason(self, feedback: Dict[str, Any]) -> str:
        """
        根据试讲反馈分类流失原因

        Returns:
            "price" / "style_mismatch" / "schedule" / "no_need" / "other"
        """
        if feedback.get("would_continue"):
            return "not_churn"

        text = (feedback.get("feedback_text") or "").lower()
        rating = feedback.get("rating") or 3

        if rating <= 2:
            if "贵" in text or "价格" in text or "太贵" in text:
                return "price"
            if any(kw in text for kw in ["风格", "严厉", "温柔", "启发", "不适合"]):
                return "style_mismatch"
            if any(kw in text for kw in ["时间", "不方便", "调课", "排课"]):
                return "schedule"

        if "不需" in text or "不需要" in text or "暂不" in text:
            return "no_need"

        return "other"

    def update_student_profile_after_churn(
        self,
        student_id: int,
        feedback: Dict[str, Any],
        churn_reason: str,
    ) -> bool:
        """
        流失后更新学生画像（影响后续推荐）
        """
        try:
            updates = {"churn_risk": min(1.0, 0.3 + 0.2)}

            # 流失原因影响后续推荐策略
            if churn_reason == "style_mismatch":
                # 风格不匹配，保留偏好但记录
                updates["notes"] = (
                    f"曾因教学风格不匹配流失（{feedback.get('feedback_text', '')}）"
                )
            elif churn_reason == "price":
                updates["notes"] = "对价格敏感，优先推荐性价比套餐"
            elif churn_reason == "no_need":
                updates["churn_risk"] = 0.6
                updates["notes"] = "暂无辅导需求"

            return self.student_service.update_student(student_id, **updates)
        except Exception as e:
            logger.error(f"更新流失画像失败: {e}")
            return False

    # ------------------- 转化消息生成 -------------------

    def generate_conversion_message(
        self,
        teacher_name: str,
        packages: List[Dict[str, Any]],
        student_name: str = "孩子",
    ) -> str:
        """生成转化推荐消息"""
        if not packages:
            return (
                f"\n顾问：{teacher_name}老师的试讲已完成！\n"
                f"如果您对老师满意，可以在「/packages」页面选购合适的套餐，"
                f"或者告诉我孩子的学科和年级，我为您推荐。\n"
            )

        lines = [
            f"\n顾问：✅ {teacher_name}老师的试讲已完成！",
            f"\n根据{student_name}的情况，为您推荐以下套餐：\n",
        ]

        for idx, pkg in enumerate(packages, 1):
            lines.append(
                f"{idx}. **{pkg['name']}**\n"
                f"   - 学科：{pkg.get('subject', '未指定')} | 年级：{pkg.get('grade_level', '未指定')}\n"
                f"   - 共 {pkg['total_lessons']} 课时，每节 {pkg['duration_minutes']} 分钟\n"
                f"   - 总价：¥{pkg['price_total']}（¥{pkg['price_per_lesson']:.0f}/课时）\n"
                f"   - 推荐理由：{pkg.get('match_reason', '综合推荐')}\n"
            )

        lines.append(
            "\n回复「套餐1」或「套餐2」即可购买。"
            "购买后我们会根据您的时段要求自动生成周固定排课。\n"
        )
        return "\n".join(lines)