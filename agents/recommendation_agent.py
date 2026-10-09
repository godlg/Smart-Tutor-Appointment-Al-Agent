"""
老师推荐 Agent（家教场景）

承接主控分到 RECOMMENDATION 意图后的对话流程。
复用 trial_processor / lesson_processor 的 LLM JSON 提取模式,
把家长口语诉求解析成结构化字段(学科 / 年级 / 风格),然后调
RecommendationService.recommend_teachers 跑打分,输出 Top-N + 理由 + ID。

设计要点:
- 可空依赖注入 context_manager (P1 统一上下文管理器约定,未注入回退旧链)
- 字段不全 → 追问,不直接拒答(避免"和正式课无关"踢回主路由的死循环重演)
- JSON 解析失败 → 全"未知"兜底,落追问分支
- 学员画像从 StudentProfileService.initialize_default_student 拿
"""

import uuid
import json
import re
from typing import AsyncGenerator, Dict, Any, Optional, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from config.model_provider import create_chat_model
from config.identity import get_current_identity
from .context.types import ContextBudgetExceeded


def _coerce_bool(value) -> bool:
    """把 JSON 里的布尔字段归一到真布尔:字符串"false"/"true"与真布尔统一。

    根因:LLM(deepseek-chat)字段提取常输出 `"info_complete": "false"`(带引号字符串),
    Python 里非空字符串恒为真 bool("false")==True,会把"信息不齐全"误判成"信息齐全",
    跳过字段追问、拿"未知"直接打分推荐。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# ----------------------------------------------------------------------
# 字段提取器(LLM → JSON,失败兜底全"未知")
# ----------------------------------------------------------------------

EXTRACTION_INSTRUCTION = (
    "你是 K12 家教平台的老师匹配助手，帮家长推荐合适的家教老师。\n"
    "请从家长输入中提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
    "{{\n"
    '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
    '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
    '  "preferred_style": "教学风格偏好（严格/温和/启发式/高压/未知）",\n'
    '  "preferred_teacher_name": "指定老师姓名（无则未知）",\n'
    '  "mode": "授课形式（线上/线下/未知）",\n'
    '  "info_complete": "true/false，subject+grade_level 都不为未知时为 true",\n'
    '  "missing_info": "缺失字段列表"\n'
    "}}\n"
    "再次强调：只输出纯 JSON。"
)


class RecommendationInputParser:
    """老师推荐需求解析器（LLM 字段提取 → JSON）。"""

    def __init__(self, llm: BaseChatModel, context_manager=None, session_id=None):
        self.llm = llm
        self.context_manager = context_manager
        self.session_id = session_id
        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            template=(
                "你是 K12 家教平台的老师匹配助手，帮家长推荐合适的家教老师。\n"
                "家长输入：{user_input}\n\n"
                "请提取以下信息为纯 JSON（不要任何 markdown 标记）：\n"
                "{{\n"
                '  "subject": "学科（数学/英语/物理/化学/语文/生物/历史/地理/政治/未知）",\n'
                '  "grade_level": "年级段（小学低年级/小学高年级/初中/高中/未知）",\n'
                '  "preferred_style": "教学风格偏好（严格/温和/启发式/高压/未知）",\n'
                '  "preferred_teacher_name": "指定老师姓名（无则未知）",\n'
                '  "mode": "授课形式（线上/线下/未知）",\n'
                '  "info_complete": "true/false，subject+grade_level 都不为未知时为 true",\n'
                '  "missing_info": "缺失字段列表"\n'
                "}}\n"
                "再次强调：只输出纯 JSON。"
            ),
        )
        self.chain = self.prompt | self.llm

    def extract(self, user_input: str) -> Dict[str, Any]:
        """同步调用 LLM 拿字段(JSON 字典);解析失败兜底全"未知"。"""
        try:
            if getattr(self, "context_manager", None) is not None:
                prompt = self.context_manager.build(
                    task=user_input, session_id=getattr(self, "session_id", None),
                    extra_contexts={"提取指令": EXTRACTION_INSTRUCTION.format()},
                )
                msg = self.llm.invoke([{"role": "user", "content": prompt}])
            else:
                msg = self.chain.invoke({"user_input": user_input})
            data = json.loads(msg.content)
        except ContextBudgetExceeded:
            raise
        except Exception:
            return {
                "subject": "未知",
                "grade_level": "未知",
                "preferred_style": "未知",
                "preferred_teacher_name": "未知",
                "mode": "未知",
                "info_complete": False,
                "missing_info": ["所有信息"],
            }
        # LLM 常把布尔字段输出成字符串("true"/"false"),归一化为真布尔,
        # 避免 bool("false")==True 把"信息不齐全"误判成"齐全"而跳过追问。
        for key in ("info_complete", "unrelated"):
            if key in data:
                data[key] = _coerce_bool(data[key])
        return data


# ----------------------------------------------------------------------
# 推荐消息构造器(纯模板,无 LLM 调用,降低延迟)
# ----------------------------------------------------------------------

class RecommendationMessageBuilder:
    """老师推荐相关的回复消息模板。"""

    MISSING_PROMPTS = {
        "subject": "请问需要辅导哪个学科？",
        "grade_level": "孩子目前哪个年级段？",
        "preferred_style": "对老师风格有偏好吗（严格/温和/启发式/高压）？",
    }

    def create_missing_questions(self, missing_info: List[str]) -> str:
        """信息不全时的追问消息。"""
        qs = [self.MISSING_PROMPTS.get(f, f"请补充{f}信息") for f in missing_info]
        return "\n助手：" + " ".join(qs) + "\n"

    def create_recommendations_message(self, subject: str, grade_level: str,
                                        recommendations: List[Dict[str, Any]]) -> str:
        """Top-N 老师推荐消息(含名字/打分/理由/ID)。"""
        if not recommendations:
            return (
                f"\n助手：抱歉，目前没有找到适合{grade_level}{subject}的老师。"
                "请稍后重试或调整条件。\n"
            )
        lines = [f"\n助手：根据您的需求（{grade_level}{subject}），为您推荐以下老师：\n"]
        for idx, r in enumerate(recommendations, 1):
            t = r["teacher"]
            score = r.get("score", 0)
            reason = r.get("reason", "")
            lines.append(
                f"{idx}. **{t.get('name', '未知')}** "
                f"(ID {t.get('id', '?')}) | 匹配度 {score:.0f}\n"
                f"   {reason}\n"
            )
        lines.append(
            "\n告诉我倾向哪位老师（回复姓名或 ID），我可以为您预约试讲。\n"
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 主控
# ----------------------------------------------------------------------

# 正向确认关键词（与 TrialAgent 一致,不靠 LLM,简单可靠）
POSITIVE_CONFIRM_WORDS = ("是", "好", "可以", "行", "yes", "ok", "OK")
# 负向确认关键词（换一位 / 拒绝）
NEGATIVE_CONFIRM_WORDS = ("不", "不要", "不选", "换", "no", "No", "不要了")


class RecommendationAgent:
    """老师推荐 Agent 主控制器。

    新增"确认状态机":家长对 Top-N 回复姓名/序号/ID/『是』/『不』时,
    不再当作新一轮推荐重新提取字段,而是沿当前会话的推荐结果推进
    (选定老师 → 引导预约试讲/正式课 / 拒绝重选)。
    """

    def __init__(self, session_id=None, context_manager=None, session_chat_history=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.context_manager = context_manager
        # 跨Agent共享历史：entry_managed=True时由入口唯一写入；
        # 旧内部无标记调用仍同步AI回复，供下游TrialAgent读取确认信息。
        self.session_chat_history = session_chat_history
        # 会话历史:记录每轮推荐结果(推荐状态按 session_id 隔离,复用 TrialAgent 模式)
        self.chats_by_session_id = {}
        self.history: Dict[str, Dict[str, Any]] = {}

        self.llm = create_chat_model(temperature=0)
        self.input_parser = RecommendationInputParser(
            self.llm, context_manager=self.context_manager, session_id=self.session_id
        )
        self.message_builder = RecommendationMessageBuilder()
        # 主控共享状态:由 set_shared_state 注入。确认试讲时置 TRIAL,供下游 TrialAgent 续接。
        # 为 __new__ 构造的测试兜底置 None。
        self.state = None

        # 业务服务懒加载:首次用到时再创建
        self._recommendation_service = None
        self._student_service = None

    # ------------------------------------------------------------------
    # 会话/推荐状态管理
    # ------------------------------------------------------------------

    @staticmethod
    def _default_history() -> Dict[str, Any]:
        """新会话的初始推荐状态。"""
        return {
            "last_recommendations": [],   # 最近一次 Top-N(list of dict)
            "subject": None,
            "grade_level": None,
            "preferred_style": None,
            "awaiting_confirmation": False,  # 是否处于确认推荐结果流程
            "confirmed_teacher": None,       # 家长选定的老师(dict,含 teacher/score/reason)
            "recommendation_declined": False,  # 上一轮被家长拒绝,需引导重选
        }

    def _ensure_history(self, session_id: Optional[str]) -> Dict[str, Any]:
        """取(或惰性创建)某会话的推荐状态字典。"""
        # 兼容 __new__ 构建(测试常用):补一个空的 history 容器
        if not hasattr(self, "history") or self.history is None:
            self.history = {}
        sid = session_id or self.session_id
        if sid not in self.history:
            self.history[sid] = self._default_history()
        return self.history[sid]

    # ------------------------------------------------------------------
    # 确认流程的辅助判定(纯规则,不引入新 LLM 调用)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_positive(input_text: str) -> bool:
        """是否正向确认('是/好/可以/行/yes/ok')。"""
        text = (input_text or "").strip()
        return any(text == w or text.startswith(w) for w in POSITIVE_CONFIRM_WORDS)

    @staticmethod
    def _is_negative(input_text: str) -> bool:
        """是否负向确认('不/不要/换/no')。"""
        text = (input_text or "").strip()
        return any(text == w or text.startswith(w) for w in NEGATIVE_CONFIRM_WORDS)

    @staticmethod
    def _match_by_teacher_id(last: List[Dict[str, Any]], target_id: int) -> Optional[Dict[str, Any]]:
        """在最近一次 Top-N 里按老师**真实 ID** 匹配;命中返回推荐记录(dict),否则 None。"""
        for r in last:
            if int(r["teacher"].get("id", -1)) == target_id:
                return r
        return None

    def _match_teacher(self, history: Dict[str, Any], user_input: str) -> Optional[Dict[str, Any]]:
        """在最近一次 Top-N 里匹配家长说的老师:
        ① 裸序号 '1'/'2'/'3' → 列表第 N 位
        ② 'ID 7' → 按老师真实 ID;'7号'/口语序号('选3号'/'3号老师')→ 先按真实 ID,
           找不到该 ID 再按列表第 N 位回退(推荐展示序号 ≠ 真实 ID)
        ③ 姓名('高远'/'张明')子串。
        命中返回该推荐记录(dict),否则 None。
        """
        last = history.get("last_recommendations") or []
        if not last:
            return None
        text = (user_input or "").strip()

        # ① 裸数字序号匹配
        if text in ("1", "2", "3") and len(last) >= int(text):
            return last[int(text) - 1]

        # ② 显式 "ID N" → 严格按老师真实 ID 匹配(不落到列表序号兜底)
        m = re.search(r"id\s*(\d+)", text, re.I)
        if m:
            return self._match_by_teacher_id(last, int(m.group(1)))

        # ③ "N号"(含 "选N号" / "N号老师" 等口语包裹)→ 先按真实 ID;
        #    该 ID 不在本轮推荐里 → 按列表第 N 位(1-based)回退匹配。
        #    例子:列表为 张明(id1)/周敏(id6)/高远(id7),"3号" 按 id3 找不到 → 列表第 3 位高远 ✓;
        #    "7号" 按 id7 命中 → 高远 ✓。
        m = re.search(r"(\d+)\s*号", text)
        if m:
            n = int(m.group(1))
            by_id = self._match_by_teacher_id(last, n)
            if by_id:
                return by_id
            if 1 <= n <= len(last):
                return last[n - 1]
            return None

        # ④ 姓名匹配(子串,避免别名/全名差异)
        for r in last:
            name = r["teacher"].get("name", "")
            if name and name in text:
                return r
        return None

    def _confirm_message(self, matched: Dict[str, Any]) -> str:
        """已选定某位老师的确认引导消息。"""
        name = matched["teacher"].get("name", "该老师")
        return (
            f"[REPLY][推荐助手]✅ 已为您选定 {name} 老师，"
            "确认预约试讲请回复『是』。\n"
        )

    async def _handle_confirmation(self, user_input: str, history: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """确认状态机:处理『选老师 / 是 / 不』三类家长回复。"""
        # 已确认具体老师 → 等『是』或『不』
        if history.get("confirmed_teacher"):
            if self._is_positive(user_input):
                name = history["confirmed_teacher"]["teacher"].get("name", "该老师")
                subject = history.get("subject", "未知")
                grade_level = history.get("grade_level", "未知")
                teacher_id = history["confirmed_teacher"]["teacher"].get("id")
                # 关键:把上下文塞进引导消息,TrialAgent 的 LLM 提取时能拿到 subject/grade_level
                history["awaiting_confirmation"] = False
                history["confirmed_teacher"] = None
                identity = get_current_identity()
                if identity is not None and not identity.legacy:
                    if self.state:
                        from config.constants import StateEnum
                        self.state.value = StateEnum.CLASSIFY
                    yield (
                        f"[REPLY][推荐助手]已为您确认 {name} 老师(学科: {subject}, "
                        f"年级: {grade_level}, 老师ID: {teacher_id})。"
                        "办理试讲预约需要先验证家长身份及学员归属；当前仅确认老师选择。\n"
                    )
                    return
                yield (
                    f"[REPLY][推荐助手]已为您确认 {name} 老师(学科: {subject}, "
                    f"年级: {grade_level}, 老师ID: {teacher_id})。"
                    "好的，为您预约试讲，请问您希望什么时候试讲？\n"
                )
                # 家长已明确确认试讲:共享状态置 TRIAL,下一轮由状态机直接续接 TrialAgent
                if self.state:
                    from config.constants import StateEnum
                    self.state.value = StateEnum.TRIAL
                return
            if self._is_negative(user_input):
                history["confirmed_teacher"] = None
                history["awaiting_confirmation"] = False
                history["recommendation_declined"] = True
                yield "[REPLY][推荐助手]好的，不选这位老师。告诉我学科和年级，我再为您重新推荐。\n"
                return
            # 同时换了人选 → 走老师匹配
            matched = self._match_teacher(history, user_input)
            if matched:
                history["confirmed_teacher"] = matched
                yield self._confirm_message(matched)
                return
            yield "[REPLY][推荐助手]您说的是哪位老师？请回复姓名、序号或 ID。\n"
            return

        # awaiting_confirmation 但尚未圈定老师。
        # 顺序关键:先尝试具体匹配(姓名/序号/ID)。这样 "好的,就选3号老师" 的 "好" 前缀
        # 不会把它误判成正向确认 → 默认选第一顺位;先匹配命中了就锁定家长明确指定的那位。
        matched = self._match_teacher(history, user_input)
        if matched:
            history["confirmed_teacher"] = matched
            yield self._confirm_message(matched)
            return
        if self._is_positive(user_input):
            # 默认选第一顺位
            default = (history.get("last_recommendations") or [None])[0]
            if default:
                history["confirmed_teacher"] = default
                yield self._confirm_message(default)
            else:
                yield "[REPLY][推荐助手]请先让我为您推荐老师:告诉我孩子的学科和年级。\n"
            return
        if self._is_negative(user_input):
            history["awaiting_confirmation"] = False
            history["confirmed_teacher"] = None
            history["recommendation_declined"] = True
            yield "[REPLY][推荐助手]好的，您不再需要推荐了。也可以告诉我新的学科/年级重新推荐。\n"
            return

        # 匹配/正/负均未命中 → 重说
        yield "[REPLY][推荐助手]您说的是哪位老师？请回复姓名、序号或 ID。\n"

    @property
    def recommendation_service(self):
        if self._recommendation_service is None:
            from services.recommendation_service import RecommendationService
            self._recommendation_service = RecommendationService()
        return self._recommendation_service

    @property
    def student_service(self):
        if self._student_service is None:
            from services.student_profile_service import StudentProfileService
            self._student_service = StudentProfileService()
        return self._student_service

    def set_shared_state(self, shared_state):
        """保存主控共享状态;确认试讲时置 TRIAL,供下游 TrialAgent 续接。"""
        self.state = shared_state

    def _record_shared_ai(self, content: str) -> None:
        """把本 Agent 写出的完整 AI 回复同步写入跨 Agent 共享历史。

        尤其「已为您确认 XX 老师(学科/年级/老师ID)」这类携带上游上下文的消息,
        必须落进共享历史,下游 TrialAgent 拿『试讲』时才能读到。未注入
        session_chat_history(或内容为空)时跳过;getattr 兜底缺属性实例。
        """
        hist = getattr(self, "session_chat_history", None)
        if (hist is None or getattr(hist, "entry_managed", False)
                or not content or not content.strip()):
            return
        hist.add_ai_message(content)

    async def run_stream(self, user_input: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """流式处理推荐请求。公共包装:转发 token 的同时,把本轮完整 AI 回复
        写入跨 Agent 共享历史(session_chat_history 注入时)。实际逻辑在
        _run_stream_impl,保持对既有调用方(run_stream)完全兼容。"""
        buf = ""
        async for token in self._run_stream_impl(user_input=user_input, session_id=session_id):
            buf += token
            yield token
        self._record_shared_ai(buf)

    async def _run_stream_impl(self, user_input: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """流式处理推荐请求(确认状态机版):
        ① 若处于确认流程(awaiting_confirmation / confirmed_teacher)→ 处理『选老师/是/不』
        ② 若上一轮被拒绝(recommendation_declined)→ 引导重选并清标志
        ③ 否则按原逻辑:字段提取 → 字段不全追问 / 全则调打分 → 输出 Top-N 并进入确认态
        任何异常 yield [REPLY] 兜底文案,不抛栈给家长。
        """
        try:
            history = self._ensure_history(session_id)
            text = (user_input or "").strip()
            # 记录本轮聊天(兼容测试用 __new__ 构造,保留 getattr 兜底)
            if not hasattr(self, "chats_by_session_id") or self.chats_by_session_id is None:
                self.chats_by_session_id = {}
            self.chats_by_session_id.setdefault(session_id or self.session_id, []).append(text)

            # ① 确认流程分支
            if history["awaiting_confirmation"] or history["confirmed_teacher"]:
                async for token in self._handle_confirmation(text, history):
                    yield token
                return

            # ② 上一轮被拒绝 → 引导重选,清 declined
            if history["recommendation_declined"]:
                history["recommendation_declined"] = False
                yield ("[REPLY][推荐助手]好的，我们重新为您推荐。"
                       "请告诉我孩子的学科和年级。\n")
                return

            # ③ 字段提取(原逻辑)
            data = self.input_parser.extract(user_input)
            subject = data.get("subject", "未知")
            grade_level = data.get("grade_level", "未知")
            preferred_style = data.get("preferred_style", "未知")
            info_complete = data.get("info_complete", False)

            # 字段不全 → 追问,不走拒答(防死循环)
            if not info_complete or subject == "未知" or grade_level == "未知":
                missing = []
                if subject == "未知":
                    missing.append("subject")
                if grade_level == "未知":
                    missing.append("grade_level")
                if preferred_style == "未知" and not missing:
                    missing.append("preferred_style")
                yield f"[REPLY][推荐助手]{self.message_builder.create_missing_questions(missing)}"
                return

            # 字段齐全 → 调打分算法
            try:
                identity = get_current_identity()
                if identity is None or identity.legacy:
                    self.student_service.initialize_default_student()
                # 匿名student_id只作记忆namespace，公开匹配不读取默认学员画像。
                recommendations = self.recommendation_service.recommend_teachers(
                    student={"grade_level": grade_level, "preferred_style": preferred_style},
                    subject=subject,
                    top_k=3,
                )
            except Exception as svc_err:
                yield f"[REPLY][推荐助手]推荐服务暂时不可用，请稍后重试（{type(svc_err).__name__}）\n"
                return

            # 记录本轮推荐结果,进入确认态(下一轮 await_history[awaiting] = True)
            history["last_recommendations"] = recommendations
            history["subject"] = subject
            history["grade_level"] = grade_level
            history["preferred_style"] = preferred_style
            history["confirmed_teacher"] = None
            history["recommendation_declined"] = False
            history["awaiting_confirmation"] = True
            # 同步通知主路由:下一轮家长回复(姓名/ID/是/不)直接转回 RecommendationAgent,
            # 不再经 LLM 分类器(避免「又来一次老师推荐请求」的体验错乱)。
            # 必须在 yield 之前置位——本轮主路由已走完分类,这条只是给下一轮的。
            # getattr 兜底:测试常用 __new__ 构造(无 __init__),缺 state 属性。
            shared_state = getattr(self, "state", None)
            if shared_state is not None:
                from config.constants import StateEnum
                shared_state.value = StateEnum.RECOMMENDATION_CONFIRM

            msg = self.message_builder.create_recommendations_message(
                subject, grade_level, recommendations
            )
            yield f"[REPLY][推荐助手]{msg}"

        except ContextBudgetExceeded:
            yield "[ERROR]上下文超过安全预算，请缩短输入后重试。"
        except Exception as e:
            yield f"[REPLY][推荐助手]抱歉，处理您的请求时出现了错误（{type(e).__name__}）。\n"
