"""
正式课 Agent 子模块（Phase 1：只支持单次课）

Phase 2 会扩展：
- 周固定系列课生成
- 调课级联处理
- 退课 + 退费计算
"""

import json
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime, timedelta
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


def _coerce_bool(value) -> bool:
    """把 JSON 里的布尔字段归一到真布尔:字符串"false"/"true"与真布尔统一。

    根因:LLM(deepseek-chat)字段提取常输出 `"unrelated": "false"`(带引号字符串),
    Python 里非空字符串恒为真 bool("false")==True,会把"无关"误判成真、踢回主路由
    形成死循环,或把"信息不全"误判为"信息齐全"跳过追问。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


class LessonInputParser:
    """正式课需求解析器

    context_manager: 可空依赖注入。为 None 时沿用旧 PromptTemplate 链(向后兼容,
    状态机的字段提取→硬规则校验链路原样保留);注入后由上下文管理器组装大输入。
    """

    def __init__(self, llm: BaseChatModel, context_manager=None):
        self.llm = llm
        self.context_manager = context_manager
        from config.time_config import time_config
        self.current_date = time_config.current_date_str()
        self.current_datetime = time_config.current_datetime_str()

        # 提取指令(不含历史/用户输入,注入路径用它作为额外上下文)
        self.extraction_instruction = (
            "你是一个 K12 家教平台的正式课预约助手，帮家长安排正式课时。\n"
            f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
            "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
            "{{\n"
            '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
            '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
            '  "start_time": "上课时间，YYYY-MM-DD HH:MM 格式",\n'
            '  "duration_minutes": "单次课时长分钟数，默认60",\n'
            '  "preferred_teacher_name": "指定老师姓名",\n'
            '  "preferred_style": "教学风格偏好",\n'
            '  "package_id": "已购套餐 ID（如无则未知）",\n'
            '  "confirmation": "对推荐老师的确认回复",\n'
            '  "info_complete": "true/false，subject+grade_level+start_time 都不为未知时为 true",\n'
            '  "unrelated": "是否与正式课预约无关",\n'
            '  "missing_info": "缺失字段列表"\n'
            "}}\n"
            "再次强调：只输出纯 JSON。"
        )

        self.prompt = PromptTemplate(
            input_variables=["history", "user_input"],
            template=(
                "你是一个 K12 家教平台的正式课预约助手，帮家长安排正式课时。\n"
                f"当前日期是{self.current_date}，当前北京时间是{self.current_datetime}。\n"
                "已知信息：{history}\n"
                "用户输入：{user_input}\n\n"
                "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
                "{{\n"
                '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
                '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
                '  "start_time": "上课时间，YYYY-MM-DD HH:MM 格式",\n'
                '  "duration_minutes": "单次课时长分钟数，默认60",\n'
                '  "preferred_teacher_name": "指定老师姓名",\n'
                '  "preferred_style": "教学风格偏好",\n'
                '  "package_id": "已购套餐 ID（如无则未知）",\n'
                '  "confirmation": "对推荐老师的确认回复",\n'
                '  "info_complete": "true/false，subject+grade_level+start_time 都不为未知时为 true",\n'
                '  "unrelated": "是否与正式课预约无关",\n'
                '  "missing_info": "缺失字段列表"\n'
                "}}\n"
                "再次强调：只输出纯 JSON。"
            ),
        )
        self.chain = self.prompt | self.llm

    def parse_stream(self, user_input: str, chat_history: InMemoryChatMessageHistory):
        chat_history.add_message(HumanMessage(content=user_input))
        history_str = "\n".join(
            [f"用户：{m.content}" if m.type == "human" else f"助手：{m.content}"
             for m in chat_history.messages]
        )

        if self.context_manager is not None:
            # 注入路径:提取指令 + 历史作为额外上下文,由上下文管理器组装大输入
            prompt = self.context_manager.build(
                task=user_input,
                extra_contexts={
                    "已知信息(历史对话)": history_str,
                    "提取指令": self.extraction_instruction,
                },
            )
            response_stream = self.llm.stream([{"role": "user", "content": prompt}])
        else:
            # 向后兼容:沿用旧 PromptTemplate 链,状态机字段提取链路保持不变
            response_stream = self.chain.stream(
                {"history": history_str, "user_input": user_input}
            )

        ai_content = ""
        for chunk in response_stream:
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            ai_content += token
            yield token
        chat_history.add_message(AIMessage(content=ai_content))
        return ai_content

    def parse_data(self, ai_content: str) -> Dict[str, Any]:
        try:
            data = json.loads(ai_content)
        except json.JSONDecodeError:
            return {
                "subject": "未知", "grade_level": "未知",
                "start_time": "未知", "duration_minutes": "60",
                "preferred_teacher_name": "未知", "preferred_style": "未知",
                "package_id": "未知", "confirmation": "未知",
                "info_complete": False, "unrelated": False,
                "missing_info": ["所有信息"],
            }
        # LLM 常把布尔字段输出成字符串("true"/"false"),归一化为真布尔,
        # 避免 bool("false")==True 把"无关"误判成正(死循环根因)。
        for key in ("unrelated", "info_complete"):
            if key in data:
                data[key] = _coerce_bool(data[key])
        return data


class LessonTeacherFinder:
    """正式课老师匹配（复用推荐算法）"""

    def __init__(self):
        pass

    def find_recommended_teachers(
        self,
        subject: str,
        grade_level: str,
        preferred_teacher_name: str = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        from services.recommendation_service import RecommendationService
        from services.teacher_service import TeacherService

        teacher_service = TeacherService()
        recommendation_service = RecommendationService()

        if preferred_teacher_name and preferred_teacher_name != "未知":
            teacher = teacher_service.get_teacher_by_name(preferred_teacher_name)
            if teacher:
                return [{"teacher": teacher, "score": 100, "reason": "您指定的老师"}]

        candidates = teacher_service.find_by_subject_and_grade(subject, grade_level)
        if not candidates:
            candidates = teacher_service.find_by_subject(subject)
        if not candidates:
            candidates = teacher_service.get_available_teachers()

        scored = []
        for t in candidates:
            score = recommendation_service.score_teacher_for_student(
                {"subjects": t.get("subjects"), "grade_levels": t.get("grade_levels"),
                 "teaching_style": t.get("teaching_style"), "highlights": t.get("highlights"),
                 "rating": t.get("rating"), "years_experience": t.get("years_experience")},
                {"grade_level": grade_level, "preferred_style": "未知"},
                subject=subject,
            )
            if score > 0:
                scored.append({"teacher": t, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]
        for item in top:
            item["reason"] = recommendation_service._build_reason(
                item["teacher"], {"grade_level": grade_level}, subject
            )
        return top


class LessonMessageBuilder:
    """正式课消息构造"""

    def __init__(self):
        self.missing_info_prompts = {
            "subject": "请问需要辅导哪个学科？",
            "grade_level": "孩子目前哪个年级段？",
            "start_time": "请问想安排什么时候上课？",
        }

    def create_recommendation_message(
        self,
        subject: str, grade_level: str, start_time: str,
        recommendations: List[Dict[str, Any]],
    ) -> str:
        if not recommendations:
            return f"\n助手：抱歉，没有找到适合{grade_level}{subject}的老师。\n"

        lines = [f"\n助手：根据您的需求（{grade_level}{subject}，{start_time}），推荐以下老师：\n"]
        for idx, r in enumerate(recommendations, 1):
            t = r["teacher"]
            lines.append(
                f"{idx}. **{t['name']}**（{t.get('gender', '')}）"
                f" | {t.get('teaching_style', '')} | "
                f"{t.get('hourly_rate', 0)}元/小时\n"
                f"   {t.get('bio', '')[:80]}\n"
            )
        lines.append(f"\n请告诉我倾向哪位老师（回复姓名或序号），确认后我会为您创建正式课。\n")
        return "\n".join(lines)

    def create_lesson_success_message(
        self,
        teacher: Dict[str, Any],
        scheduled_at: datetime,
        duration_minutes: int,
        price: float,
    ) -> str:
        time_str = scheduled_at.strftime("%Y-%m-%d %H:%M")
        return (
            f"\n助手：✅ 正式课已为您预约！\n\n"
            f"📚 **课程信息**\n"
            f"- 老师：**{teacher['name']}**（{teacher.get('gender', '')}）\n"
            f"- 时间：{time_str}\n"
            f"- 时长：{duration_minutes} 分钟\n"
            f"- 单价：{price:.0f} 元\n"
            f"- 教学风格：{teacher.get('teaching_style', '')}\n\n"
            f"📌 **温馨提示**\n"
            f"- 上课前 1 小时会有短信提醒\n"
            f"- 如需调课，请提前 24 小时联系\n"
            f"- 如果计划长期学习，建议购买套餐更划算\n"
        )

    def create_missing_info_questions(self, missing_info: List[str]) -> str:
        questions = [self.missing_info_prompts.get(field, f"请补充{field}信息") for field in missing_info]
        return "\n助手：" + " ".join(questions) + "\n"

    def create_unrelated_message(self) -> str:
        return "\n助手：这个问题和正式课无关，您是想试讲、咨询套餐，还是查询进度？\n"

    def create_parse_error_message(self) -> str:
        return "\n助手：抱歉没理解您的话，请重新描述一下。\n"

    def create_conflict_message(self, teacher_name: str) -> str:
        return f"\n助手：抱歉，{teacher_name}老师在该时段已被预约，请换个时间或选其他老师。\n"


class LessonProcessor:
    """正式课流程主处理器（Phase 1：单次课）"""

    def __init__(self, llm: BaseChatModel, context_manager=None):
        self.llm = llm
        # 可空统一上下文管理器,透传给输入解析器
        self.input_parser = LessonInputParser(llm, context_manager=context_manager)
        self.teacher_finder = LessonTeacherFinder()
        self.message_builder = LessonMessageBuilder()

        from services.lesson_service import LessonService
        from services.student_profile_service import StudentProfileService
        from services.teacher_service import TeacherService

        self.lesson_service = LessonService()
        self.student_service = StudentProfileService()
        self.teacher_service = TeacherService()

    def update_history_from_data(self, lesson_history: Dict[str, Any], data: Dict[str, Any]) -> bool:
        if lesson_history.get("awaiting_confirmation"):
            return self._handle_confirmation(lesson_history, data)

        for key in ["subject", "grade_level", "preferred_teacher_name", "preferred_style", "package_id"]:
            if data.get(key) and data[key] != "未知":
                lesson_history[key] = data[key]

        if data.get("start_time") and data["start_time"] != "未知":
            lesson_history["start_time"] = data["start_time"]

        if data.get("duration_minutes") and data["duration_minutes"] != "未知":
            lesson_history["duration_minutes"] = data["duration_minutes"]

        required = ["subject", "grade_level", "start_time"]
        return all(
            lesson_history.get(f) and lesson_history[f] != "未知"
            for f in required
        )

    def _handle_confirmation(self, lesson_history: Dict[str, Any], data: Dict[str, Any]) -> bool:
        confirmation = data.get("confirmation", "").lower()
        positive = any(p in confirmation for p in ["是", "好", "可以", "行", "yes", "ok"])
        negative = any(p in confirmation for p in ["不", "不要", "no", "换"])

        if positive and not negative:
            lesson_history["confirmed_teacher"] = lesson_history.get("recommended_teacher")
            lesson_history["awaiting_confirmation"] = False
            return True
        elif negative:
            lesson_history["recommendation_declined"] = True
            lesson_history["awaiting_confirmation"] = False
            return True
        return False

    async def handle_unrelated_request(self, user_input, unrelated_callback, state) -> AsyncGenerator[str, None]:
        yield "[REPLY][正式课助手]和正式课无关，已交给主路由\n"
        if unrelated_callback:
            try:
                result = await unrelated_callback(user_input)
                if hasattr(result, '__aiter__'):
                    async for token in result:
                        yield token
                else:
                    yield result
            except Exception:
                yield self.message_builder.create_unrelated_message()
        else:
            yield self.message_builder.create_unrelated_message()

    async def handle_complete_lesson(
        self, lesson_history: Dict[str, Any], session_id: str
    ) -> AsyncGenerator[str, None]:
        if lesson_history.get("recommendation_declined"):
            yield f"[REPLY][正式课助手]{self.message_builder.create_unrelated_message()}"
            lesson_history.pop("recommendation_declined", None)
            lesson_history.pop("recommended_teacher", None)
            return

        if lesson_history.get("confirmed_teacher"):
            teacher = lesson_history["confirmed_teacher"]
            reply = await self._create_lesson(teacher, lesson_history, session_id)
            yield f"[REPLY][正式课助手]{reply}"
            lesson_history.pop("confirmed_teacher", None)
            lesson_history.pop("recommended_teacher", None)
            lesson_history.pop("awaiting_confirmation", None)
            return

        if lesson_history.get("awaiting_confirmation"):
            yield f"[REPLY][正式课助手]\n请回复「是」或「不」明确选择\n"
            return

        # 老师匹配
        subject = lesson_history.get("subject")
        grade_level = lesson_history.get("grade_level")
        preferred_name = lesson_history.get("preferred_teacher_name")
        start_time_str = lesson_history.get("start_time")
        duration = int(lesson_history.get("duration_minutes", "60"))

        from config.time_config import time_config
        start_time = time_config.parse_datetime(start_time_str)
        if not start_time:
            yield f"[REPLY][正式课助手]无法识别时间「{start_time_str}」，请换个说法\n"
            return

        recommendations = self.teacher_finder.find_recommended_teachers(
            subject=subject, grade_level=grade_level,
            preferred_teacher_name=preferred_name, top_k=3,
        )

        if not recommendations:
            yield f"[REPLY][正式课助手]{self.message_builder.create_recommendation_message(subject, grade_level, start_time_str, [])}"
            return

        lesson_history["recommended_teacher"] = recommendations[0]["teacher"]
        lesson_history["recommendations"] = recommendations
        lesson_history["start_time_parsed"] = start_time
        lesson_history["duration_minutes_int"] = duration
        lesson_history["awaiting_confirmation"] = True

        msg = self.message_builder.create_recommendation_message(
            subject, grade_level, start_time_str, recommendations
        )
        yield f"[REPLY][正式课助手]{msg}"

    async def _create_lesson(
        self,
        teacher: Dict[str, Any],
        lesson_history: Dict[str, Any],
        session_id: str,
    ) -> str:
        start_time = lesson_history["start_time_parsed"]
        duration = lesson_history["duration_minutes_int"]
        package_id = lesson_history.get("package_id")
        if package_id == "未知":
            package_id = None

        student = self.student_service.initialize_default_student()
        price = teacher.get("hourly_rate", 0) * (duration / 60.0)

        success, result = self.lesson_service.create_lesson(
            teacher_id=teacher["id"],
            student_id=student["id"],
            scheduled_at=start_time,
            lesson_type="regular",
            subject=lesson_history.get("subject"),
            duration_minutes=duration,
            status="confirmed",
            package_id=package_id,
            price=price,
        )

        if success:
            return self.message_builder.create_lesson_success_message(
                teacher, start_time, duration, price
            )
        else:
            # 可能是冲突
            return self.message_builder.create_conflict_message(teacher["name"]) + f"（{result}）"

    async def handle_incomplete_info(self, data: Dict[str, Any], lesson_history: Dict[str, Any]) -> AsyncGenerator[str, None]:
        missing = []
        for field in ["subject", "grade_level", "start_time"]:
            if not lesson_history.get(field) or lesson_history[field] == "未知":
                missing.append(field)

        reply = self.message_builder.create_missing_info_questions(missing)
        yield f"[THOUGHT][正式课助手]信息不完整，缺少：{', '.join(missing)}"
        yield f"[REPLY][正式课助手]{reply}"