"""
试讲反馈处理

按 TUTOR_DESIGN.md §5.3 实现：
- 试讲反馈结构化收集
- 评分 + 文本反馈 + 是否继续 + 推荐补充学科
- 与 Lesson.status 联动更新
- 反馈结果驱动 conversion_logic
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TrialFeedbackCollector:
    """试讲反馈收集器"""

    def __init__(self, lesson_service=None, student_service=None):
        self._lesson_service = lesson_service
        self._student_service = student_service

    @property
    def lesson_service(self):
        if self._lesson_service is None:
            from services.lesson_service import LessonService
            self._lesson_service = LessonService()
        return self._lesson_service

    @property
    def student_service(self):
        if self._student_service is None:
            from services.student_profile_service import StudentProfileService
            self._student_service = StudentProfileService()
        return self._student_service

    def submit_feedback(
        self,
        lesson_id: int,
        parent_id: int,
        rating: int,
        would_continue: bool,
        feedback_text: str = None,
        recommended_subjects: list = None,
    ) -> Tuple[bool, str, Optional[int]]:
        """
        提交试讲反馈

        Returns:
            (success, message, feedback_id)
        """
        if rating < 1 or rating > 5:
            return False, "评分必须在 1-5 之间", None

        # 1. 验证 lesson 存在且是试讲
        lesson = self.lesson_service.get_lesson(lesson_id)
        if not lesson:
            return False, "课时不存在", None
        if lesson.get("lesson_type") != "trial":
            return False, "该课时不是试讲课", None

        # 2. 写反馈
        try:
            feedback_id = self.lesson_service.add_trial_feedback(
                lesson_id=lesson_id,
                parent_id=parent_id,
                rating=rating,
                would_continue=would_continue,
                feedback_text=feedback_text,
                recommended_subjects=recommended_subjects,
            )
        except Exception as e:
            logger.error(f"写试讲反馈失败: {e}")
            return False, f"反馈提交失败：{e}", None

        # 3. 更新 Lesson 状态为已完成
        self.lesson_service.update_lesson(lesson_id, status="completed")

        # 4. 更新学生画像
        try:
            student = self.student_service.get_student(lesson["student_id"])
            if student:
                # 评分影响续费信号
                if would_continue and rating >= 4:
                    self.student_service.update_engagement(
                        student["id"], upsell_signal="strong"
                    )
                elif would_continue and rating >= 3:
                    self.student_service.update_engagement(
                        student["id"], upsell_signal="weak"
                    )
                else:
                    self.student_service.update_engagement(
                        student["id"], upsell_signal="none"
                    )
        except Exception as e:
            logger.warning(f"更新学生画像失败（不影响反馈提交）: {e}")

        return True, "反馈提交成功", feedback_id

    def get_feedback_summary(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        """获取试讲反馈摘要"""
        feedback = self.lesson_service.get_trial_feedback(lesson_id)
        if not feedback:
            return None

        return {
            "feedback_id": feedback["id"],
            "lesson_id": lesson_id,
            "rating": feedback["rating"],
            "would_continue": feedback["would_continue"],
            "feedback_text": feedback.get("feedback_text"),
            "recommended_subjects": feedback.get("recommended_subjects") or [],
            "created_at": feedback.get("created_at"),
        }

    def get_pending_feedback_lessons(self) -> list:
        """获取待填写反馈的试讲课时"""
        return self.lesson_service.get_pending_trials()

    def build_feedback_prompt_message(self, teacher_name: str) -> str:
        """构造引导家长填写反馈的消息"""
        return (
            f"\n顾问：{teacher_name} 老师的试讲已完成！📝\n\n"
            f"请花 1 分钟告诉我您的感受：\n"
            f"1. 总体评分（1-5 星）\n"
            f"2. 是否希望孩子继续跟这位老师学习？（是/否）\n"
            f"3. 任何具体感受或建议\n\n"
            f"回复格式示例：「5 分，是，孩子很喜欢老师的讲解方式」\n"
        )