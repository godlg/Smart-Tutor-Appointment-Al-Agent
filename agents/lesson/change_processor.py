"""
调课 / 退课对话处理器

背景：
- RescheduleHandler / CancelHandler 是「面向 lesson_id」的执行层，REST 接口
  （api/lesson.py）可以直接传 lesson_id 调用。
- 但聊天入口拿到的是自然语言（"把周三的课改到周五"），没有 lesson_id。

本模块补上中间这一层：
  自然语言 → 定位家长名下的目标课时 → 解析新时间/退课原因 → 调用已有 Handler

依赖方向遵守 5 层单向依赖：agents → services / db，不反向 import。
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from .cancel_handler import CancelHandler
from .reschedule_handler import RescheduleHandler, RescheduleMode

logger = logging.getLogger(__name__)

# 一次最多向家长展示 / 供 LLM 选择的待上课时数量
MAX_CANDIDATE_LESSONS = 10


class LessonChangeParser:
    """调课 / 退课需求解析器

    把家长的自然语言 + 名下待上课时列表交给 LLM，抽取出：
    目标课时 ID、新时间、调整范围、退课原因。
    """

    def __init__(self, llm: BaseChatModel, context_manager=None):
        self.llm = llm
        self.context_manager = context_manager
        from config.time_config import time_config
        self.current_date = time_config.current_date_str()
        self.current_datetime = time_config.current_datetime_str()

        # 提取指令(不含 lessons/用户输入,注入路径用它作为额外上下文)
        self.extraction_instruction = (
            "你是一个 K12 家教平台的课程管家，负责处理家长的{intent_desc}请求。\n"
            f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
            "家长名下待上的课时如下（只能从这些课时里选，不要编造 ID）：\n"
            "{lessons}\n"
            "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
            "{{\n"
            '  "target_lesson_id": "家长想调整/取消的课时 ID，无法确定就填未知",\n'
            '  "new_time": "调课的新时间，YYYY-MM-DD HH:MM 格式，退课或未提及则填未知",\n'
            '  "scope": "only_this 表示只改这一次，this_and_following 表示这次及之后每周都改，默认 only_this",\n'
            '  "reason": "退课/调课原因，未提及则填未知"\n'
            "}}\n"
            "注意：家长说「周三那节」「明天的课」时，请对照上面列表的实际日期来匹配；\n"
            "如果说法含糊、对应多节课或一节都对不上，target_lesson_id 必须填未知。\n"
            "再次强调：只输出纯 JSON。"
        )

        self.prompt = PromptTemplate(
            input_variables=["intent_desc", "lessons", "user_input"],
            template=(
                "你是一个 K12 家教平台的课程管家，负责处理家长的{intent_desc}请求。\n"
                f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
                "家长名下待上的课时如下（只能从这些课时里选，不要编造 ID）：\n"
                "{lessons}\n"
                "家长本次输入：{user_input}\n\n"
                "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
                "{{\n"
                '  "target_lesson_id": "家长想调整/取消的课时 ID，无法确定就填未知",\n'
                '  "new_time": "调课的新时间，YYYY-MM-DD HH:MM 格式，退课或未提及则填未知",\n'
                '  "scope": "only_this 表示只改这一次，this_and_following 表示这次及之后每周都改，默认 only_this",\n'
                '  "reason": "退课/调课原因，未提及则填未知"\n'
                "}}\n"
                "注意：家长说「周三那节」「明天的课」时，请对照上面列表的实际日期来匹配；\n"
                "如果说法含糊、对应多节课或一节都对不上，target_lesson_id 必须填未知。\n"
                "再次强调：只输出纯 JSON。"
            ),
        )
        self.chain = self.prompt | self.llm

    async def parse(self, user_input: str, lessons_text: str, intent: str) -> Dict[str, Any]:
        """调用 LLM 解析，失败时返回全「未知」，由上层走澄清分支"""
        intent_desc = "调课" if intent == "reschedule" else "退课"
        try:
            if self.context_manager is not None:
                # 注入路径:提取指令(填充意图与课时列表)作为额外上下文
                instruction = self.extraction_instruction.format(
                    intent_desc=intent_desc, lessons=lessons_text
                )
                prompt = self.context_manager.build(
                    task=user_input,
                    extra_contexts={"提取指令": instruction},
                )
                message = await self.llm.ainvoke(
                    [{"role": "user", "content": prompt}]
                )
            else:
                # 向后兼容:沿用旧 PromptTemplate 链
                message = await self.chain.ainvoke({
                    "intent_desc": intent_desc,
                    "lessons": lessons_text,
                    "user_input": user_input,
                })
            content = message.content if hasattr(message, "content") else str(message)
            return self.parse_data(content)
        except Exception as e:
            logger.warning(f"调课/退课意图解析失败: {e}")
            return self._empty_data()

    def parse_data(self, ai_content: str) -> Dict[str, Any]:
        try:
            data = json.loads(ai_content.strip())
            if not isinstance(data, dict):
                return self._empty_data()
            return data
        except (json.JSONDecodeError, AttributeError):
            return self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        return {
            "target_lesson_id": "未知",
            "new_time": "未知",
            "scope": RescheduleMode.ONLY_THIS,
            "reason": "未知",
        }


class LessonChangeMessageBuilder:
    """调课 / 退课消息构造"""

    def create_no_lesson_message(self, intent: str) -> str:
        action = "调整" if intent == "reschedule" else "取消"
        return (
            f"\n助手：您名下暂时没有待上的课时，所以没有可{action}的课程。\n"
            f"如果需要预约新课程，直接告诉我学科、年级和时间就可以。\n"
        )

    def create_lesson_list(self, lessons: List[Dict[str, Any]]) -> str:
        """把待上课时渲染成家长可读的列表（同时用于 LLM 匹配）"""
        lines = []
        for lesson in lessons:
            scheduled_at = lesson.get("scheduled_at")
            time_str = (
                scheduled_at.strftime("%Y-%m-%d %H:%M")
                if hasattr(scheduled_at, "strftime")
                else str(scheduled_at)
            )
            series_mark = "（周固定系列课）" if lesson.get("series_id") else ""
            lines.append(
                f"- ID {lesson.get('id')}：{time_str} "
                f"{lesson.get('subject') or ''} "
                f"{lesson.get('teacher_name') or ''}老师 "
                f"{lesson.get('duration_minutes') or 0}分钟{series_mark}"
            )
        return "\n".join(lines)

    def create_ambiguous_target_message(
        self, intent: str, lessons: List[Dict[str, Any]]
    ) -> str:
        action = "调整" if intent == "reschedule" else "取消"
        return (
            f"\n助手：没能确定您要{action}哪一节课。您名下待上的课时有：\n"
            f"{self.create_lesson_list(lessons)}\n\n"
            f"请告诉我具体是哪一节（回复课时 ID 或准确的上课时间）。\n"
        )

    def create_missing_new_time_message(self, lesson: Dict[str, Any]) -> str:
        scheduled_at = lesson.get("scheduled_at")
        time_str = (
            scheduled_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(scheduled_at, "strftime")
            else str(scheduled_at)
        )
        return (
            f"\n助手：您要调整的是 {time_str} 的{lesson.get('subject') or ''}课。\n"
            f"请问希望改到什么时间？（例如：本周五下午 4 点）\n"
        )

    def create_reschedule_success_message(
        self, lesson: Dict[str, Any], new_time_str: str, detail: str
    ) -> str:
        return (
            f"\n助手：✅ 调课成功！\n\n"
            f"📚 **课程信息**\n"
            f"- 课时 ID：{lesson.get('id')}\n"
            f"- 学科：{lesson.get('subject') or ''}\n"
            f"- 新时间：{new_time_str}\n"
            f"- 处理结果：{detail}\n\n"
            f"📌 已同步给老师，上课前 1 小时仍会有提醒。\n"
        )

    def create_reschedule_failed_message(self, detail: str) -> str:
        return (
            f"\n助手：抱歉，这次调课没有成功：{detail}\n"
            f"您可以换一个时间再试，或者告诉我其它方便的时段。\n"
        )

    def create_cancel_success_message(
        self, lesson: Dict[str, Any], refund_info: Optional[Dict[str, Any]]
    ) -> str:
        scheduled_at = lesson.get("scheduled_at")
        time_str = (
            scheduled_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(scheduled_at, "strftime")
            else str(scheduled_at)
        )
        lines = [
            "\n助手：✅ 退课已办理。\n",
            "📚 **已取消的课程**",
            f"- 课时 ID：{lesson.get('id')}",
            f"- 学科：{lesson.get('subject') or ''}",
            f"- 原定时间：{time_str}\n",
        ]
        if refund_info:
            lines.append("💰 **退费明细**")
            lines.append(f"- 原价：{refund_info.get('original_price', 0):.0f} 元")
            lines.append(f"- 退款金额：{refund_info.get('refund_amount', 0):.0f} 元")
            lines.append(f"- 适用政策：{refund_info.get('policy', '')}\n")
        lines.append("如需重新安排时间，随时告诉我。\n")
        return "\n".join(lines)

    def create_cancel_failed_message(self, detail: str) -> str:
        return f"\n助手：抱歉，这次退课没有成功：{detail}\n"


class LessonChangeProcessor:
    """调课 / 退课流程处理器

    对话入口的调课/退课都走这里：先定位课时，再复用
    RescheduleHandler / CancelHandler 执行。
    """

    def __init__(self, llm: BaseChatModel, context_manager=None):
        self.llm = llm
        # 可空统一上下文管理器,透传给调课/退课解析器
        self.parser = LessonChangeParser(llm, context_manager=context_manager)
        self.message_builder = LessonChangeMessageBuilder()
        self.reschedule_handler = RescheduleHandler()
        self.cancel_handler = CancelHandler()

        from services.lesson_service import LessonService
        from services.student_profile_service import StudentProfileService
        from services.teacher_service import TeacherService

        self.lesson_service = LessonService()
        self.student_service = StudentProfileService()
        self.teacher_service = TeacherService()

    async def handle_change_request(
        self, user_input: str, intent: str
    ) -> AsyncGenerator[str, None]:
        """调课/退课统一入口，intent 取值 reschedule / cancel"""
        intent_desc = "调课" if intent == "reschedule" else "退课"
        yield f"[THOUGHT][正式课助手]识别为{intent_desc}请求，正在查询您名下的待上课时。"

        lessons = self._load_upcoming_lessons()
        if not lessons:
            yield f"[REPLY][正式课助手]{self.message_builder.create_no_lesson_message(intent)}"
            return

        lessons_text = self.message_builder.create_lesson_list(lessons)
        data = await self.parser.parse(user_input, lessons_text, intent)

        target = self._match_target_lesson(lessons, data.get("target_lesson_id"))
        if not target:
            # 定位不到就澄清，绝不「猜一节课」去改或取消
            yield f"[REPLY][正式课助手]{self.message_builder.create_ambiguous_target_message(intent, lessons)}"
            return

        if intent == "reschedule":
            async for token in self._handle_reschedule(target, data):
                yield token
        else:
            async for token in self._handle_cancel(target, data):
                yield token

    def _load_upcoming_lessons(self) -> List[Dict[str, Any]]:
        """查询默认学生名下待上的课时，并补上老师姓名便于家长辨认"""
        try:
            student = self.student_service.initialize_default_student()
            lessons = self.lesson_service.get_student_upcoming_lessons(
                student["id"], limit=MAX_CANDIDATE_LESSONS
            )
        except Exception as e:
            logger.warning(f"查询待上课时失败: {e}")
            return []

        for lesson in lessons:
            try:
                teacher = self.teacher_service.get_teacher_by_id(lesson["teacher_id"])
                lesson["teacher_name"] = teacher.get("name") if teacher else ""
            except Exception:
                lesson["teacher_name"] = ""
        return lessons

    def _match_target_lesson(
        self, lessons: List[Dict[str, Any]], target_id: Any
    ) -> Optional[Dict[str, Any]]:
        """只接受能在候选列表里对上的 ID，避免 LLM 编造 ID 误伤其它课时"""
        if target_id is None or target_id == "未知":
            return None
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            return None
        return next((l for l in lessons if l.get("id") == target_id), None)

    async def _handle_reschedule(
        self, lesson: Dict[str, Any], data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        from config.time_config import time_config

        new_time_str = data.get("new_time")
        new_time = (
            time_config.parse_datetime(new_time_str)
            if new_time_str and new_time_str != "未知"
            else None
        )
        if not new_time:
            yield f"[REPLY][正式课助手]{self.message_builder.create_missing_new_time_message(lesson)}"
            return

        mode = (
            RescheduleMode.THIS_AND_FOLLOWING
            if data.get("scope") == RescheduleMode.THIS_AND_FOLLOWING
            else RescheduleMode.ONLY_THIS
        )
        success, message = self.reschedule_handler.reschedule(
            lesson_id=lesson["id"], new_time=new_time, mode=mode
        )

        if success:
            reply = self.message_builder.create_reschedule_success_message(
                lesson, time_config.format_datetime(new_time), message
            )
        else:
            reply = self.message_builder.create_reschedule_failed_message(message)
        yield f"[REPLY][正式课助手]{reply}"

    async def _handle_cancel(
        self, lesson: Dict[str, Any], data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        reason = data.get("reason")
        if reason == "未知":
            reason = None

        success, message, refund_info = self.cancel_handler.cancel_lesson(
            lesson_id=lesson["id"], reason=reason
        )

        if success:
            reply = self.message_builder.create_cancel_success_message(lesson, refund_info)
        else:
            reply = self.message_builder.create_cancel_failed_message(message)
        yield f"[REPLY][正式课助手]{reply}"
