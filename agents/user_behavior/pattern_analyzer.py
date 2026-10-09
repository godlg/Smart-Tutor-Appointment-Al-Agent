"""
家长行为模式分析器（家教场景）

核心功能：
1. 分析家长偏好的老师
2. 分析常用学科 / 时段 / 时长
3. 判断是否需要回访邀请
4. 生成个性化回访 / 续费消息
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging


class PatternAnalyzer:
    """家长行为模式分析器"""

    def __init__(self, behavior_service=None):
        self.behavior_service = behavior_service
        self.logger = logging.getLogger(__name__)

    @property
    def behavior_db(self):
        if hasattr(self, "behavior_service") and self.behavior_service:
            return self.behavior_service.user_behavior_repo
        return None

    def analyze_user_preferences(self, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """分析家长偏好：偏好的老师、常用学科、常用时长"""
        try:
            if self.behavior_service:
                appointments = self.behavior_service.get_user_behaviors(
                    user_id=user_id, action_type="lesson"
                )
            else:
                appointments = self.behavior_db.get_user_behaviors(
                    user_id=user_id, action_type="lesson"
                )

            if not appointments:
                return None

            teacher_counts = {}
            subject_counts = {}
            duration_counts = {}

            for appt in appointments:
                data = appt.get("action_data", {})

                teacher_id = appt.get("teacher_id")
                if teacher_id:
                    teacher_counts[teacher_id] = teacher_counts.get(teacher_id, 0) + 1

                subject = data.get("subject")
                if subject:
                    subject_counts[subject] = subject_counts.get(subject, 0) + 1

                duration = data.get("duration_minutes")
                if duration:
                    duration_counts[duration] = duration_counts.get(duration, 0) + 1

            favorite_teacher = (
                max(teacher_counts, key=teacher_counts.get) if teacher_counts else None
            )
            favorite_subject = (
                max(subject_counts, key=subject_counts.get) if subject_counts else None
            )
            favorite_duration = (
                max(duration_counts, key=duration_counts.get) if duration_counts else None
            )

            return {
                "favorite_teacher_id": favorite_teacher,
                "favorite_subject": favorite_subject,
                "favorite_duration": favorite_duration,
                "total_lessons": len(appointments),
                "last_lesson_date": appointments[0]["created_at"] if appointments else None,
            }
        except Exception as e:
            self.logger.error(f"分析家长偏好失败: {str(e)}")
            return None

    def should_send_return_reminder(self, user_id: str = "default_user", days_threshold: int = 30) -> bool:
        """判断是否应该发送回访提醒"""
        try:
            preferences = self.analyze_user_preferences(user_id)
            if not preferences or preferences["total_lessons"] < 2:
                return False

            last_lesson = preferences["last_lesson_date"]
            if not last_lesson:
                return False

            if isinstance(last_lesson, str):
                last_lesson = datetime.fromisoformat(last_lesson.replace("Z", "+00:00"))

            days_since_last = (datetime.now() - last_lesson).days
            return days_since_last >= days_threshold
        except Exception as e:
            self.logger.error(f"判断回访提醒失败: {str(e)}")
            return False

    def generate_return_message(self, user_id: str = "default_user") -> Optional[str]:
        """生成个性化回访消息"""
        try:
            preferences = self.analyze_user_preferences(user_id)
            if not preferences:
                return "您好！孩子最近学习怎么样？要不要预约一节正式课？"

            teacher_id = preferences.get("favorite_teacher_id")
            subject = preferences.get("favorite_subject", "辅导")
            duration = preferences.get("favorite_duration", 60)

            if teacher_id:
                from services.teacher_service import TeacherService
                teacher = TeacherService().get_teacher_by_id(teacher_id)
                teacher_name = teacher.get("name", "熟悉的老师") if teacher else "熟悉的老师"
                message = (
                    f"您好！{teacher_name}老师最近有档空，之前孩子上过"
                    f"{subject}的辅导，"
                )
                if duration:
                    message += f"按习惯的{duration}分钟，"
                message += "要不要再约一节？"
            else:
                message = f"您好！孩子最近学习怎么样？要不要预约一节{subject}辅导？"
                if duration:
                    message += f"按习惯的{duration}分钟怎么样？"
            return message
        except Exception as e:
            self.logger.error(f"生成回访消息失败: {str(e)}")
            return "您好！孩子最近学习怎么样？要不要预约一节正式课？"