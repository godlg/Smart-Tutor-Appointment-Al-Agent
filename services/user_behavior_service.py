"""
家长行为服务层
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import Counter

from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)


class UserBehaviorService:
    """家长行为服务类"""

    def __init__(self, db_path: str = 'sqlite:///data/smart_appointment.db'):

        self.db_router = DatabaseRouter(db_path)
        self.user_behavior_repo = self.db_router.user_behavior

    def record_behavior(
        self,
        user_id: str,
        action_type: str,
        action_data: Dict[str, Any] = None,
        teacher_id: int = None,
        session_id: str = "default_session",
    ) -> bool:
        """记录家长行为"""
        try:
            behavior_id = self.user_behavior_repo.record_behavior(
               
                user_id=user_id,
                action_type=action_type,
                action_data=action_data,
                teacher_id=teacher_id,
                session_id=session_id,
            )
            if behavior_id:
                logger.info(
                    f"家长行为记录成功：用户={user_id}, 行为={action_type}, ID={behavior_id}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"记录家长行为失败：{e}")
            return False

    def get_user_behaviors(
        self, user_id: str, action_type: str = None, days_back: int = None
    ) -> List[Dict[str, Any]]:
        return self.user_behavior_repo.get_user_behaviors(user_id, action_type, days_back)

    def get_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        return self.user_behavior_repo.get_user_preferences(user_id)

    def update_user_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: str,
        confidence_score: int = 1,
    ) -> bool:
        return self.user_behavior_repo.update_user_preference(
            user_id, preference_type, preference_value, confidence_score
        )

    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """分析家长行为模式"""
        behaviors = self.get_user_behaviors(user_id, days_back=30)
        if not behaviors:
            return {"pattern": "no_data", "recommendation": "需要更多数据"}

        lesson_behaviors = [b for b in behaviors if b.get("action_type") == "lesson"]
        freq_analysis = self._analyze_frequency(lesson_behaviors)
        preferred_teacher = self._analyze_preferred_teacher(lesson_behaviors)
        time_preference = self._analyze_time_preference(lesson_behaviors)

        return {
            "pattern": "active_user" if len(lesson_behaviors) > 2 else "occasional_user",
            "frequency_analysis": freq_analysis,
            "preferred_teacher": preferred_teacher,
            "time_preference": time_preference,
            "total_lessons": len(lesson_behaviors),
            "analysis_period_days": 30,
        }

    def _analyze_frequency(self, lesson_behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not lesson_behaviors:
            return {"frequency": "no_lessons", "days_between": 0}

        if len(lesson_behaviors) < 2:
            return {"frequency": "single_lesson", "days_between": 0}

        dates = []
        for behavior in lesson_behaviors:
            if "created_at" in behavior:
                try:
                    date = datetime.fromisoformat(behavior["created_at"].replace("Z", "+00:00"))
                    dates.append(date)
                except Exception:
                    continue

        if len(dates) < 2:
            return {"frequency": "insufficient_data", "days_between": 0}

        dates.sort()
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        if avg_interval < 7:
            frequency = "very_frequent"
        elif avg_interval < 14:
            frequency = "frequent"
        elif avg_interval < 30:
            frequency = "regular"
        else:
            frequency = "occasional"

        return {"frequency": frequency, "days_between": avg_interval}

    def _analyze_preferred_teacher(self, lesson_behaviors: List[Dict[str, Any]]) -> Optional[str]:
        teacher_counts = {}
        for behavior in lesson_behaviors:
            teacher_id = behavior.get("teacher_id")
            if teacher_id:
                teacher_counts[teacher_id] = teacher_counts.get(teacher_id, 0) + 1

        if not teacher_counts:
            return None
        most_frequent = max(teacher_counts, key=teacher_counts.get)
        return most_frequent if teacher_counts[most_frequent] > 1 else None

    def _analyze_time_preference(self, lesson_behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        hours = []
        weekdays = []

        for behavior in lesson_behaviors:
            action_data = behavior.get("action_data", {})
            if isinstance(action_data, dict) and "scheduled_at" in action_data:
                try:
                    scheduled = datetime.fromisoformat(action_data["scheduled_at"])
                    hours.append(scheduled.hour)
                    weekdays.append(scheduled.weekday())
                except Exception:
                    continue

        if not hours:
            return {"preferred_hour": None, "preferred_weekday": None}

        hour_counter = Counter(hours)
        weekday_counter = Counter(weekdays)

        preferred_hour = hour_counter.most_common(1)[0][0] if hour_counter else None
        preferred_weekday = weekday_counter.most_common(1)[0][0] if weekday_counter else None

        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        return {
            "preferred_hour": preferred_hour,
            "preferred_weekday": weekday_names[preferred_weekday] if preferred_weekday is not None else None,
            "hour_distribution": dict(hour_counter),
            "weekday_distribution": dict(weekday_counter),
        }