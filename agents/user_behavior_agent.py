"""
家长行为分析代理（家教场景）

核心功能：
1. 分析家长偏好（最常预约的老师 / 学科 / 时段 / 时长）
2. 判断回访 / 续费时机
3. 生成个性化回访 / 续费提醒消息
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from dotenv import load_dotenv
from config.model_provider import create_chat_model
from .user_behavior import PatternAnalyzer, BehaviorRecorder, PreferenceManager

load_dotenv()


class UserBehaviorAgent:
    """家长行为分析代理（家教版）"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self._user_behavior_service = None
        self.llm = self._initialize_llm()

        # 通过 Service 层访问
        from services.user_behavior_service import UserBehaviorService
        self.behavior_service = UserBehaviorService()

        self.pattern_analyzer = PatternAnalyzer(self.behavior_service)
        self.behavior_recorder = BehaviorRecorder(self.behavior_service)
        self.preference_manager = PreferenceManager(self.behavior_service)

    @property
    def user_behavior_service(self):
        if self._user_behavior_service is None:
            from services.user_behavior_service import UserBehaviorService
            self._user_behavior_service = UserBehaviorService()
        return self._user_behavior_service

    def _initialize_llm(self):
        return create_chat_model(temperature=0.7)

    def record_behavior(
        self,
        action_type: str,
        action_data: Dict[str, Any],
        teacher_id: int = None,
        session_id: str = "default_session",
    ) -> bool:
        """记录家长行为"""
        try:
            return self.user_behavior_service.record_behavior(
                user_id="default_user",
                action_type=action_type,
                action_data=action_data,
                teacher_id=teacher_id,
                session_id=session_id,
            )
        except Exception as e:
            self.logger.error(f"记录用户行为失败: {str(e)}")
            try:
                return self.behavior_recorder.record_behavior(
                    action_type=action_type,
                    action_data=action_data,
                    teacher_id=teacher_id,
                    session_id=session_id,
                )
            except Exception as fallback_error:
                self.logger.error(f"回退方法也失败: {str(fallback_error)}")
                return False

    def get_user_analysis(self, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """获取家长行为分析数据"""
        try:
            preferences = self.pattern_analyzer.analyze_user_preferences(user_id)
            if not preferences:
                return None

            last_lesson = preferences.get("last_lesson_date") or preferences.get("last_appointment_date")
            days_since_last = None
            if last_lesson:
                if isinstance(last_lesson, str):
                    last_lesson = datetime.fromisoformat(
                        last_lesson.replace("Z", "+00:00")
                    )
                days_since_last = (datetime.now() - last_lesson).days

            return {
                "favorite_teacher_id": preferences.get("favorite_teacher_id"),
                "favorite_subject": preferences.get("favorite_subject"),
                "favorite_duration": preferences.get("favorite_duration"),
                "total_lessons": preferences.get("total_lessons"),
                "days_since_last_lesson": days_since_last,
                "should_send_reminder": self.pattern_analyzer.should_send_return_reminder(user_id),
            }
        except Exception as e:
            self.logger.error(f"获取用户分析失败: {str(e)}")
            return None

    def generate_reminder_message(self, user_id: str = "default_user") -> Optional[str]:
        """生成回访提醒消息（默认模板）"""
        try:
            return self.pattern_analyzer.generate_return_message(user_id)
        except Exception as e:
            self.logger.error(f"生成提醒消息失败: {str(e)}")
            return None

    async def generate_personalized_reminder(
        self,
        user_id: str = "default_user",
        available_times: list = None,
    ) -> Optional[str]:
        """使用 LLM 生成个性化提醒消息"""
        try:
            self.logger.info(f"开始生成个性化提醒，用户ID: {user_id}")

            analysis = self.get_user_analysis(user_id)
            if not analysis or not analysis.get("favorite_teacher_id"):
                return (
                    "尊敬的家长，您好！孩子最近学习怎么样？要不要预约一节正式课？"
                )

            from services.teacher_service import TeacherService
            teacher = TeacherService().get_teacher_by_id(analysis["favorite_teacher_id"])
            teacher_name = teacher.get("name", "熟悉的老师") if teacher else "熟悉的老师"
            subjects = teacher.get("subjects", []) if teacher else []
            subject = subjects[0] if subjects else (analysis.get("favorite_subject") or "")

            duration = analysis.get("favorite_duration") or 60

            times_text = (
                "、".join([t["formatted"] for t in (available_times or [])[:3]])
                if available_times
                else "暂时没有空闲时间"
            )

            prompt = (
                "请为家教平台家长生成一条温暖、亲切的回访 / 续费提醒消息。\n\n"
                "信息：\n"
                f"- 偏好的老师：{teacher_name}\n"
                f"- 学科：{subject}\n"
                f"- 常用时长：{duration} 分钟\n"
                f"- 老师空闲时段：{times_text}\n\n"
                "要求：\n"
                "1. 语气亲切自然，不要生硬推销\n"
                "2. 提到老师的名字和学科\n"
                "3. 如果有空闲时间，自然地提及\n"
                "4. 控制在 80 字以内\n"
                "5. 直接输出消息内容，不要任何标记\n"
            )

            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"LLM 生成个性化提醒失败: {type(e).__name__}: {str(e)}")
            return self.generate_reminder_message(user_id)

    async def get_reminder_with_schedule(
        self, user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """获取包含时间安排的完整提醒信息"""
        try:
            analysis = self.get_user_analysis(user_id)
            if not analysis or not analysis.get("favorite_teacher_id"):
                message = await self.generate_personalized_reminder(user_id, [])
                return {"message": message, "teacher_available_times": []}

            teacher_id = analysis["favorite_teacher_id"]
            duration = analysis.get("favorite_duration") or 60

            from services.lesson_service import LessonService
            lesson_svc = LessonService()
            today = datetime.now()
            available_times: List[Dict[str, str]] = []

            current_hour = today.hour
            start_hour = max(current_hour + 1, 9)

            for hour in range(start_hour, 22):
                check_time = today.replace(hour=hour, minute=0, second=0, microsecond=0)
                end_time = check_time + timedelta(minutes=duration)
                if not lesson_svc.lesson_repo.check_teacher_conflict(
                    teacher_id, check_time, end_time
                ):
                    available_times.append({
                        "date": today.strftime("%Y-%m-%d"),
                        "time": f"{hour}:00",
                        "formatted": f"今天{hour}:00",
                    })

            if not available_times:
                tomorrow = today + timedelta(days=1)
                for hour in range(9, 22):
                    check_time = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                    end_time = check_time + timedelta(minutes=duration)
                    if not lesson_svc.lesson_repo.check_teacher_conflict(
                        teacher_id, check_time, end_time
                    ):
                        available_times.append({
                            "date": tomorrow.strftime("%Y-%m-%d"),
                            "time": f"{hour}:00",
                            "formatted": f"明天{hour}:00",
                        })
                        if len(available_times) >= 3:
                            break

            message = await self.generate_personalized_reminder(user_id, available_times)
            return {"message": message, "teacher_available_times": available_times}

        except Exception as e:
            self.logger.error(f"获取提醒信息失败: {type(e).__name__}: {str(e)}")
            return {
                "message": "尊敬的家长，您好！系统暂时无法查询老师时段，请稍后再试。",
                "teacher_available_times": [],
            }