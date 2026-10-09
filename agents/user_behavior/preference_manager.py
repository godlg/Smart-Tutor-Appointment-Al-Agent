"""
偏好管理器（家教场景）

负责管理家长的偏好数据：
- 老师偏好
- 学科偏好
- 时段偏好
- 教学风格偏好
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging


class PreferenceManager:
    """家长偏好管理器"""

    def __init__(self, behavior_service=None):
        self.behavior_service = behavior_service
        self.logger = logging.getLogger(__name__)

    @property
    def behavior_db(self):
        if hasattr(self, "behavior_service") and self.behavior_service:
            return self.behavior_service.user_behavior_repo
        return None

    def update_preferences_from_lesson(self, action_data: Dict[str, Any], teacher_id: int = None):
        """从课时数据更新家长偏好"""
        try:
            if teacher_id:
                self.update_teacher_preference(teacher_id)

            if action_data.get("scheduled_at"):
                self.update_time_preference(action_data["scheduled_at"])

            if action_data.get("duration_minutes"):
                self.update_duration_preference(action_data["duration_minutes"])

            if action_data.get("subject"):
                self.update_subject_preference(action_data["subject"])

            if action_data.get("preferred_style"):
                self.update_style_preference(action_data["preferred_style"])
        except Exception as e:
            self.logger.error(f"更新家长偏好失败: {str(e)}")

    def update_teacher_preference(self, teacher_id: int):
        try:
            self.behavior_db.update_user_preference("teacher", str(teacher_id))
            self.logger.info(f"更新老师偏好: {teacher_id}")
        except Exception as e:
            self.logger.error(f"更新老师偏好失败: {str(e)}")

    def update_time_preference(self, scheduled_at: str):
        try:
            dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
            hour = dt.hour
            if 6 <= hour < 12:
                period = "上午"
            elif 12 <= hour < 18:
                period = "下午"
            else:
                period = "晚上"
            self.behavior_db.update_user_preference("time_period", period)
            self.logger.info(f"更新时间偏好: {period}")
        except Exception as e:
            self.logger.error(f"更新时间偏好失败: {str(e)}")

    def update_duration_preference(self, duration: Any):
        try:
            self.behavior_db.update_user_preference("duration", str(duration))
            self.logger.info(f"更新时长偏好: {duration}")
        except Exception as e:
            self.logger.error(f"更新时长偏好失败: {str(e)}")

    def update_subject_preference(self, subject: str):
        try:
            self.behavior_db.update_user_preference("subject", subject)
            self.logger.info(f"更新学科偏好: {subject}")
        except Exception as e:
            self.logger.error(f"更新学科偏好失败: {str(e)}")

    def update_style_preference(self, style: str):
        try:
            self.behavior_db.update_user_preference("teaching_style", style)
            self.logger.info(f"更新教学风格偏好: {style}")
        except Exception as e:
            self.logger.error(f"更新教学风格偏好失败: {str(e)}")

    def get_user_preferences(self) -> Dict[str, Any]:
        try:
            return self.behavior_db.get_user_preferences()
        except Exception as e:
            self.logger.error(f"获取用户偏好失败: {str(e)}")
            return {}

    def get_preferred_teacher_id(self) -> Optional[int]:
        try:
            preferences = self.get_user_preferences()
            teacher_id = preferences.get("teacher")
            return int(teacher_id) if teacher_id else None
        except Exception as e:
            self.logger.error(f"获取偏好老师ID失败: {str(e)}")
            return None

    def get_preferred_time_period(self) -> Optional[str]:
        try:
            preferences = self.get_user_preferences()
            return preferences.get("time_period")
        except Exception as e:
            self.logger.error(f"获取偏好时间段失败: {str(e)}")
            return None

    def get_preferred_subject(self) -> Optional[str]:
        try:
            preferences = self.get_user_preferences()
            return preferences.get("subject")
        except Exception as e:
            self.logger.error(f"获取偏好学科失败: {str(e)}")
            return None

    def get_preference_summary(self) -> Dict[str, Any]:
        try:
            preferences = self.get_user_preferences()
            summary = {
                "has_teacher_preference": bool(preferences.get("teacher")),
                "has_time_preference": bool(preferences.get("time_period")),
                "has_subject_preference": bool(preferences.get("subject")),
                "has_duration_preference": bool(preferences.get("duration")),
                "preference_count": len([v for v in preferences.values() if v]),
            }
            if preferences.get("teacher"):
                summary["preferred_teacher_id"] = int(preferences["teacher"])
            if preferences.get("time_period"):
                summary["preferred_time"] = preferences["time_period"]
            if preferences.get("subject"):
                summary["preferred_subject"] = preferences["subject"]
            if preferences.get("duration"):
                summary["preferred_duration"] = preferences["duration"]
            return summary
        except Exception as e:
            self.logger.error(f"获取偏好摘要失败: {str(e)}")
            return {}

    def clear_preference(self, preference_type: str):
        try:
            self.behavior_db.update_user_preference(preference_type, None)
            self.logger.info(f"清除偏好: {preference_type}")
        except Exception as e:
            self.logger.error(f"清除偏好失败: {str(e)}")

    def clear_all_preferences(self):
        try:
            for pref_type in ["teacher", "time_period", "duration", "subject", "teaching_style"]:
                self.clear_preference(pref_type)
            self.logger.info("清除所有家长偏好")
        except Exception as e:
            self.logger.error(f"清除所有偏好失败: {str(e)}")