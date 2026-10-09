"""
家长行为分析 API
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/user-behavior", tags=["家长行为分析"])
router_underscore = APIRouter(prefix="/api/user_behavior", tags=["家长行为分析"])


class UserAnalysisResponse(BaseModel):
    favorite_teacher_id: Optional[int] = None
    favorite_teacher_name: Optional[str] = None
    favorite_subject: Optional[str] = None
    favorite_duration: Optional[int] = None
    total_lessons: int = 0
    days_since_last_lesson: Optional[int] = None
    should_send_reminder: bool = False


async def get_user_analysis(user_id: str = "default_user") -> UserAnalysisResponse:
    try:
        from agents.user_behavior_agent import UserBehaviorAgent

        agent = UserBehaviorAgent()
        analysis = agent.get_user_analysis(user_id)

        if not analysis:
            return UserAnalysisResponse()

        teacher_name = None
        if analysis.get("favorite_teacher_id"):
            from services.teacher_service import TeacherService
            teacher = TeacherService().get_teacher_by_id(analysis["favorite_teacher_id"])
            if teacher:
                teacher_name = teacher.get("name")

        return UserAnalysisResponse(
            favorite_teacher_id=analysis.get("favorite_teacher_id"),
            favorite_teacher_name=teacher_name,
            favorite_subject=analysis.get("favorite_subject"),
            favorite_duration=analysis.get("favorite_duration"),
            total_lessons=analysis.get("total_lessons", 0),
            days_since_last_lesson=analysis.get("days_since_last_lesson"),
            should_send_reminder=analysis.get("should_send_reminder", False),
        )
    except Exception as e:
        import logging
        logging.error(f"获取家长分析数据失败: {e}")
        return UserAnalysisResponse()


@router.get("/analysis", response_model=UserAnalysisResponse)
async def get_default_user_analysis():
    return await get_user_analysis("default_user")


@router.get("/dashboard_data", response_model=UserAnalysisResponse)
async def get_dashboard_data():
    return await get_user_analysis("default_user")


@router_underscore.get("/dashboard_data", response_model=UserAnalysisResponse)
async def get_dashboard_data_underscore():
    return await get_user_analysis("default_user")


class ReminderRequest(BaseModel):
    user_id: str = "default_user"


class ReminderResponse(BaseModel):
    message: str
    teacher_available_times: Optional[list] = None


@router.post("/send-reminder", response_model=ReminderResponse, summary="发送回访提醒")
async def send_reminder(request: ReminderRequest):
    try:
        from agents.user_behavior_agent import UserBehaviorAgent

        agent = UserBehaviorAgent()
        result = await agent.get_reminder_with_schedule(request.user_id)

        return ReminderResponse(
            message=result["message"],
            teacher_available_times=result["teacher_available_times"],
        )
    except Exception as e:
        import logging
        logging.error(f"生成回访提醒失败: {e}")
        return ReminderResponse(
            message="尊敬的家长，您好！系统暂时无法查询老师时段，请稍后再试。",
            teacher_available_times=[],
        )


@router.get("/recommendations", summary="拉取未发送的家长推荐")
async def list_recommendations(user_id: str = "default_user", mark_sent: bool = False):
    """
    返回该家长名下仍未发送的 UserRecommendation 列表。
    推荐系统(后台调度器)持续往 user_recommendations 落库,
    家长中心或聊天窗口通过本端点拉取并展示。

    mark_sent=true 时,把拉到的这批标记为已发送(避免重复展示)。
    """
    from db.db_router import DatabaseRouter
    from agents.user_behavior_agent import UserBehaviorAgent

    try:
        db = DatabaseRouter()
        # user_id 在家长中心场景下是 'student_<id>' 或 'default_user'
        # 兼容: 传 default_user 时把名下所有 student_<id> 的推荐也合进来
        user_ids = [user_id]
        if user_id == "default_user":
            for s in db.students.get_all_students():
                user_ids.append(f"student_{s['id']}")

        all_pending = []
        for uid in user_ids:
            all_pending.extend(db.user_behavior.get_pending_recommendations(uid))

        if mark_sent:
            for r in all_pending:
                try:
                    db.user_behavior.mark_recommendation_sent(r["id"])
                except Exception:
                    pass

        return {"success": True, "count": len(all_pending), "data": all_pending}
    except Exception as e:
        import logging
        logging.error(f"拉取推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))