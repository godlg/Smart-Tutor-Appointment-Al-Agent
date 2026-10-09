"""
试讲 Agent 入口

继承并复用 Appointment Agent 的状态机思想，但专注于试讲转化流程
"""

import uuid
from typing import AsyncGenerator, Dict, Any
from langchain_core.chat_history import InMemoryChatMessageHistory
from config.model_provider import create_chat_model
from .trial_processor import TrialProcessor


class TrialAgent:
    """试讲 Agent 主控制器"""

    def __init__(self, session_id=None, unrelated_callback=None, context_manager=None,
                 session_chat_history=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.unrelated_callback = unrelated_callback
        self.state = None
        # 统一上下文管理器(可空):注入后字段提取由它组装输入,否则回退旧逻辑
        self.context_manager = context_manager
        # 跨Agent共享历史：entry_managed=True时由入口唯一写入，本Agent只读；
        # 旧内部调用无标记时仍同步AI回复。局部chat_history只保存提取诊断。
        self.session_chat_history = session_chat_history

        self.llm = create_chat_model(temperature=0)

        self.processor = TrialProcessor(
            self.llm, context_manager=self.context_manager, session_id=self.session_id,
            session_chat_history=self.session_chat_history,
        )

        self.chats_by_session_id = {}
        self.chat_history = self._get_chat_history(self.session_id)
        self.reset()

    def _build_context_manager(self):
        """组装试讲 Agent 用统一上下文管理器。

        价值点:conversation(多轮追问上下文)+ structured_output(JSON 提取约束)。
        返回的 manager 可注入 TrialAgent(context_manager=...)。
        """
        from agents.context import (
            ContextManager,
            SystemPromptContext,
            ConversationContext,
            StructuredOutputContext,
            ToolContext,
        )
        history = getattr(self, "session_chat_history", None)
        manager = ContextManager([
            SystemPromptContext(),
            ConversationContext(history_provider=(lambda sid: history) if history is not None else None),
            StructuredOutputContext(),
            ToolContext(),
        ])
        manager.session_id = self.session_id
        return manager

    def _get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        chat_history = self.chats_by_session_id.get(session_id)
        if chat_history is None:
            chat_history = InMemoryChatMessageHistory()
            self.chats_by_session_id[session_id] = chat_history
        return chat_history

    def reset(self):
        self.trial_history = {
            "subject": None,
            "grade_level": None,
            "start_time": None,
            "duration_minutes": "30",
            "student_situation": None,
            "preferred_teacher_name": None,
            "preferred_style": None,
            # 递归防护:同一实例连续两次把输入判为"无关"踢回主路由时,主路由重分类
            # 仍命中本站 → 二次直接兜底不再调 callback,防止无限递归。
            "unrelated_escalations": 0,
        }
        self.finished = False
        self.chat_history.clear()

    def set_shared_state(self, shared_state):
        self.state = shared_state

    @staticmethod
    def _looks_like_extraction_json(content: str) -> bool:
        """判定是否为"字段提取模型"的纯 JSON 输出。

        上游确认消息是自然语言(如"已为您确认 周敏 老师(学科: 数学...)"),
        而 TrialAgent 自己的字段提取结果是纯 JSON 对象。识别纯 JSON 可以把它
        从"上游上下文"兜底候选中排除,避免被误当成上游确认(反向用例)。
        """
        if not content:
            return False
        stripped = (content or "").strip()
        return bool(stripped.startswith("{") and stripped.endswith("}") and ":" in stripped)

    def _record_shared_ai(self, content: str) -> None:
        """把本 Agent 写出的完整 AI 回复同步写入跨 Agent 共享历史。

        未注入 session_chat_history(或内容为空)时直接跳过,保持旧行为。
        用 getattr 兜底,兼容测试里经 object.__new__ 直接构造(缺该属性)的实例。
        """
        hist = getattr(self, "session_chat_history", None)
        if (hist is None or getattr(hist, "entry_managed", False)
                or not content or not content.strip()):
            return
        hist.add_ai_message(content)

    def _last_ai_context(self) -> str:
        """从跨 Agent 共享历史中找回"上游已确认"的学科/年级/老师上下文。

        优先匹配携带「学科:」/「老师ID:」字段的 AI 消息——这类消息是上游
        推荐 Agent(RecommendationAgent)把确认结果写入共享历史的,能精准定位
        学科/年级/老师,避免『试讲』这类短输入只剩主语、字段全未知。
        若没有(或被 TrialAgent 自己的 LLM JSON 提取输出占据),则兜底取最近一条
        "非 JSON 提取输出"的 AI 消息;仍未找到/未注入共享历史时返回空串(原行为)。
        """
        hist = getattr(self, "session_chat_history", None)
        if hist is None:
            return ""
        msgs = hist.messages
        # ① 优先:携带学科/年级/老师ID 的上游确认消息
        confirmed = [
            m for m in msgs
            if m.type == "ai"
            and ("学科:" in m.content or "老师ID:" in m.content)
        ]
        if confirmed:
            return f"上游上下文: {confirmed[-1].content}"
        # ② 兜底:取任意最近"非字段提取 JSON"的 AI 消息(排除自己的 LLM JSON 响应)
        fallback = [
            m for m in msgs
            if m.type == "ai" and not self._looks_like_extraction_json(m.content)
        ]
        if fallback:
            return f"上游上下文: {fallback[-1].content}"
        return ""

    @staticmethod
    def _extract_confirmed_context(context_str: str) -> Dict[str, Any]:
        """从上游确认消息(『已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)』)
        解析出 subject / grade_level / preferred_teacher_name / teacher_id 四个字段。

        用法:在 TrialAgent run_stream 开头,若 trial_history.subject 为 None 但
        上游已确认时,调用本函数把字段回填到 trial_history,避免 TrialAgent
        重新跑 find_recommended_teachers 又列一遍老师。
        返回 dict 缺字段时为 None;解析失败/无上游消息返回空 dict。
        """
        import re as _re
        if not context_str:
            return {}
        out: Dict[str, Any] = {}
        # 学科:优先在「学科: 数学」里抓,其次「X 数学 辅导需求」里的学科枚举
        m = _re.search(r"学科[:：]\s*([\u4e00-\u9fa5A-Za-z]+)", context_str)
        if m:
            out["subject"] = m.group(1).strip()
        # 年级:优先「年级: 初中」,其次「X 年级」「X 年级段」
        m = _re.search(r"年级[:：]?\s*([\u4e00-\u9fa5]+)", context_str)
        if m:
            out["grade_level"] = m.group(1).strip()
        # 老师ID:必须出现「老师ID: 7」这种格式才抓(避免误匹配年份/数字)
        m = _re.search(r"老师ID[:：]\s*(\d+)", context_str)
        if m:
            out["teacher_id"] = int(m.group(1))
        # 老师姓名:「已为您确认 周敏 老师」/「已为您选定 周敏 老师」
        m = _re.search(r"(?:已为您(?:确认|选定))\s*([\u4e00-\u9fa5]{2,4})\s*老师", context_str)
        if m:
            out["preferred_teacher_name"] = m.group(1).strip()
        return out

    async def run_stream(self, user_input=None):
        """试讲流式处理"""
        if user_input is None:
            user_input = input("家长：")

        buf = ""
        async for token in self._run_stream_impl(user_input=user_input):
            buf += token
            yield token
        # 把本轮完整 AI 回复同步写入跨 Agent 共享历史(仅注入 session_chat_history 时)
        self._record_shared_ai(buf)

    async def _run_stream_impl(self, user_input):
        """run_stream 的实际实现(纯生成器),被公共包装方法转发并记录共享历史。"""

        # 分支 0(新增):用户对上轮『direct trial 引导消息』回『确认/是/好』+
        # 有 pending_trial → 真正创建 Lesson(TRIAL),不调 LLM、不走 TrialInputParser。
        # 设计目的:把『确认请回复』这条引导消息的最终落库闭环在 TrialAgent 内,
        # 不再依赖 TaskClassifier LLM(避免『确认』被误判成 recommendation 等)。
        # trial_confirm_route spec:同一 TrialAgent 实例跨多轮共享 trial_history,
        # pending_trial 由 handle_complete_trial 在上一轮存进来,本轮判断后清空。
        pending = self.trial_history.get("pending_trial")
        if pending:
            # 局部导入避免顶层拉服务层,符合 5 层单向依赖
            from .trial_processor import _is_simple_confirm, _is_convert_intent
            if _is_simple_confirm(user_input):
                async for token in self._confirm_pending_trial(pending):
                    yield token
                return
            # 转化意图:家长在待确认阶段直接表达『试讲满意,想转正式课』
            # (如『试讲觉得不错,就要他俩』)。先真正创建本次试讲 Lesson(TRIAL),
            # 再进入转化引导 awaiting_convert。不走 TrialInputParser,避免 LLM
            # 把这种话抽成 unrelated 报『没理解』(trial-convert-followup 根因)。
            if _is_convert_intent(user_input):
                async for token in self._handle_convert_intent(pending):
                    yield token
                return
            # 输入不是简单确认(如『改成周三晚上 8 点』『不对,我想选张老师』):
            # 不走 confirm 分支,但 pending_trial 仍保留——让 TrialInputParser
            # 自然按正常 trial 流程重抽(用户想改时间/改老师就走正常链路)。
            # 只有家长输入被判定为『简单确认』后才清掉 pending_trial。

        # 分支 convert-continue:家长已表达『试讲满意』(awaiting_convert=True)后的
        # 转化续轮。处理三类输入:继续/满意(推荐套餐)、选套餐(转正式课)、流失否定(兜底)。
        # 与 pending_time_query 同理:不调 LLM、不走 TrialInputParser。
        # 分支顺序:pending_trial(含 convert 判断)→ awaiting_convert → pending_time_query → 正常解析。
        if self.trial_history.get("awaiting_convert"):
            async for token in self._handle_awaiting_convert(user_input):
                yield token
            return

        # 分支 0b(时间追问):上一轮 handle_complete_trial 时间解析失败(或缺时间)
        # 时设了 pending_time_query → 本轮家长补充具体时间去接住,不调 LLM、
        # 不走 TrialInputParser(避免重跑字段提取又丢时间)。
        # 命中否定词(不约了/取消/算了/换时间)则清标志走正常流程,让家长重新表达。
        pending_time = self.trial_history.get("pending_time_query")
        if pending_time:
            from .trial_processor import _is_time_negation
            if _is_time_negation(user_input):
                self.trial_history.pop("pending_time_query", None)
                self.trial_history.pop("pending_date_hint", None)
                self.trial_history.pop("pending_time_attempts", None)
                # 清标志后继续往下走正常 TrialInputParser 解析
            else:
                async for token in self._handle_time_followup(user_input):
                    yield token
                return

        ai_content = ""
        # 上下文字符串:优先取上一条 AI 输出携带的学科/年级/老师信息
        context = self._last_ai_context()
        # 回填上游已确认的字段(仅在 trial_history 对应字段为 None 时回填,
        # 不覆盖家长后续可能补充的新输入)。根因:推荐 agent 已在前面轮次确认
        # 了『学科/年级/老师』,若 TrialAgent 不主动回填,trial_history 全是 None,
        # LLM 提取出来还是『未知』,最终 handle_complete_trial 又跑 find_recommended_teachers
        # 把前面选好的老师重新列一遍让家长选——这正是用户报的 bug。
        confirmed = self._extract_confirmed_context(context)
        if confirmed:
            for key in ("subject", "grade_level", "preferred_teacher_name"):
                if confirmed.get(key) and not self.trial_history.get(key):
                    self.trial_history[key] = confirmed[key]
            if confirmed.get("teacher_id") and not self.trial_history.get("teacher_id"):
                self.trial_history["teacher_id"] = confirmed["teacher_id"]
        for token in self.processor.input_parser.parse_stream(
            user_input, self.chat_history, context=context
        ):
            ai_content += token

        try:
            data = self.processor.input_parser.parse_data(ai_content)
            # B 兜底:LLM 字段提取对『这个星期周五晚上』这类相对时间表达常直接返回
            # start_time=未知,导致试讲流程永远走『信息不完整,缺少:start_time』。
            # 这里在字段层做一次确定性兜底:LLM 没抽到时间时,直接用
            # time_config.parse_datetime(build)解析家长原始输入(相对/绝对都行),
            # 把换算后的绝对时间补回 data,喂给状态机,不再依赖 LLM 做相对时间换算。
            # (parse_datetime 内部已增强支持 这个星期X/这周X/周五晚上 等,见 config/time_config.py)
            from .trial_processor import _fallback_start_time_from_input
            _fallback_start_time_from_input(data, user_input)
            self.finished = self.processor.update_history_from_data(self.trial_history, data)

            # 无关请求
            if data.get("unrelated", False) and not self.trial_history.get("awaiting_confirmation"):
                if self.state:
                    from config.constants import StateEnum
                    self.state.value = StateEnum.CLASSIFY
                escalated = self.trial_history.get("unrelated_escalations", 0)
                if escalated >= 1:
                    # 防御:这个输入已经踢回过一次主路由(主路由重分类仍命中本次)。
                    # 二次命中 → 不再调 callback(避免无限递归),直接兜底并重置计数。
                    self.trial_history["unrelated_escalations"] = 0
                    async for token in self.processor.handle_unrelated_request(
                        user_input, None, self.state
                    ):
                        yield token
                    return
                self.trial_history["unrelated_escalations"] = 1
                async for token in self.processor.handle_unrelated_request(
                    user_input, self.unrelated_callback, self.state
                ):
                    yield token
                return

            # 完整流程
            if self.finished:
                async for token in self.processor.handle_complete_trial(
                    self.trial_history, self.session_id,
                    chat_history=self.chat_history.messages,
                ):
                    yield token

                # 时间追问:handle_complete_trial 时间解析失败(或缺时间)时设了
                # pending_time_query。此时不能 reset trial_history(会把标志清掉),
                # 直接 return,让下一轮入口的 pending_time_query 分支接住家长补的时间。
                if self.trial_history.get("pending_time_query"):
                    return

                # 新增:handle_complete_trial 走 direct trial 分支并 yield 了
                # 『确认请回复』引导消息后,会在 trial_history 存 pending_trial。
                # 这种情况下需要把 state 切到 TRIAL_CONFIRM,
                # 让主路由在下一轮『确认』时跳过分类器直接转给 TrialAgent,
                # 而不是让 TaskClassifier LLM 误判『确认』+『老师』为 recommendation。
                # (跟 recommendation_confirm 的处理同型)
                if self.trial_history.get("pending_trial"):
                    if self.state:
                        from agents.task_classification.state_manager import FlowState
                        self.state.value = FlowState.TRIAL_CONFIRM
                    return  # pending_trial 在 trial_history 里,state 已切,不下一次重置

                if not self.trial_history.get("awaiting_confirmation"):
                    self._reset_state_after_trial()
                return

            # 信息不完整
            async for token in self.processor.handle_incomplete_info(
                data, self.trial_history, user_input=user_input
            ):
                yield token

        except Exception as e:
            yield self.processor.message_builder.create_parse_error_message()

    async def _confirm_pending_trial(self, pending: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """家长对上轮『direct trial 引导消息』回『确认/是/好』时真正创建 Lesson(TRIAL)。

        与 _create_trial 的区别:
        - _create_trial 由 handle_complete_trial 内部调用,需要 teacher dict + conflict check,
          已被 trial_processor 走『直接预约消息』分支跳过。
        - _confirm_pending_trial 由 TrialAgent.run_stream 入口直接调用,
          只用 pending 里的 id + time,无需 conflict check(同一会话同一时间,不会冲突),
          但走 LessonService 拿到完整 lesson_id(用于 yield 成功消息 + 后续查看)。

    不引入新 LLM 调用;调用方已确保 user_input 是简单确认。
        无论成功失败都清掉 pending_trial + 重置 state(防呆,避免下一轮再次走 confirm 分支)。
        """
        from .trial_processor import _create_pending_trial_lesson

        success, lesson_id_or_err = _create_pending_trial_lesson(
            self.processor,
            teacher_id=pending["teacher_id"],
            time_iso=pending["time_iso"],
            duration_minutes=pending.get("duration_minutes", 30),
            subject=pending.get("subject"),
        )
        # 无论成功失败都清掉 pending_trial(防呆,避免下一轮再触发 confirm 分支)
        self.trial_history.pop("pending_trial", None)
        # 顺手清理 pending_trial 残留的辅助字段(已用不上,避免污染下一轮 trial_history)
        for key in ("confirmed_teacher_id", "confirmed_teacher", "start_time_parsed",
                    "duration_minutes_int"):
            self.trial_history.pop(key, None)

        if success:
            yield (
                f"[REPLY][试讲顾问]🎉 试讲已成功预约!\n"
                f"老师: {pending.get('teacher_name', '')}\n"
                f"时间: {pending.get('time_str', '')}\n"
                f"课程 ID: {lesson_id_or_err}\n"
                f"课前 1 小时会发送提醒。\n"
            )
        else:
            yield f"[REPLY][试讲顾问]试讲预约失败: {lesson_id_or_err}\n"

        # 重置 trial_history + state 归 CLASSIFY(防止下一轮再走 TRIAL_CONFIRM 分支)
        self._reset_state_after_trial()

    async def _handle_convert_intent(self, pending: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """家长在上轮『试讲预约引导』后回『试讲满意/想继续上』时,处理『转正式课』意向。

        参照 _confirm_pending_trial:先用 pending 真正创建本次试讲 Lesson(TRIAL),
        成功则 yield 试讲预约成功消息;再进入转化引导(awaiting_convert),提示家长
        试讲结束后告知感受,由系统推荐套餐并转正式课。不引入新 LLM 调用。

        与 _confirm_pending_trial 的区别:创建成功后【不】reset trial_history——
        要保留 awaiting_convert / convert_teacher_id / convert_subject / convert_lesson_id,
        并切 state 到 TRIAL_CONVERT,让下一轮家长的表达直接进本 Agent 的
        awaiting_convert 分支(不被 TaskClassifier 判 unrelated)。
        """
        from .trial_processor import _create_pending_trial_lesson

        success, lesson_id_or_err = _create_pending_trial_lesson(
            self.processor,
            teacher_id=pending["teacher_id"],
            time_iso=pending["time_iso"],
            duration_minutes=pending.get("duration_minutes", 30),
            subject=pending.get("subject"),
        )
        # 无论成败都清掉 pending_trial(防呆,与 _confirm_pending_trial 一致)
        self.trial_history.pop("pending_trial", None)
        for key in ("confirmed_teacher_id", "confirmed_teacher", "start_time_parsed",
                    "duration_minutes_int"):
            self.trial_history.pop(key, None)

        if success:
            yield (
                f"[REPLY][试讲顾问]🎉 试讲已成功预约!\n"
                f"老师: {pending.get('teacher_name', '')}\n"
                f"时间: {pending.get('time_str', '')}\n"
                f"课程 ID: {lesson_id_or_err}\n"
            )
            # 进入转化态:记录用于推荐套餐的试讲老师/学科/课时
            self.trial_history["awaiting_convert"] = True
            self.trial_history["convert_teacher_id"] = pending.get("teacher_id")
            self.trial_history["convert_subject"] = pending.get("subject")
            self.trial_history["convert_lesson_id"] = lesson_id_or_err
            yield (
                "[REPLY][试讲顾问]试讲结束后告诉我感受（如:试讲不错,想继续）,"
                "我为您推荐合适套餐并转正式课。\n"
            )
        else:
            yield f"[REPLY][试讲顾问]试讲预约失败: {lesson_id_or_err}\n"

        # 切到 TRIAL_CONVERT:主路由下一轮家长表达继续时跳过分类器直接转本 Agent。
        if self.state:
            from agents.task_classification.state_manager import FlowState
            self.state.value = FlowState.TRIAL_CONVERT

    async def _handle_awaiting_convert(self, user_input) -> AsyncGenerator[str, None]:
        """家长在『试讲满意』引导后的转化续轮处理。

        三类输入(确定性分支,不调 LLM):
        1. 流失否定(先不上了/不要/再考虑)→ 引导兜底 + 清 awaiting_convert
        2. 套餐选择(1 / 第一个 / 套餐1 / 套餐名)→ 调用转正式课逻辑(创建系列课)
        3. 继续/满意(试讲不错,想继续)→ 推荐正式课套餐(Top-3)
        """
        from .trial_processor import (
            _is_convert_intent, _is_positive_continue, _is_declined_convert_intent,
        )

        s = (user_input or "").strip()
        if not s:
            yield "[REPLY][试讲顾问]回复套餐序号/名称购买;或告诉我试讲感受(如:试讲不错,想继续)。\n"
            return
        # 1. 流失否定优先:避免把『先不上了』当套餐选择/正向表达
        if _is_declined_convert_intent(s):
            self._clear_convert_state()
            yield "[REPLY][试讲顾问]好的,有需要随时找我。\n"
            return
        # 2. 套餐选择:按已推荐的 convert_packages 匹配序号/名称
        pkg = self._match_package_selection(s)
        if pkg is not None:
            async for token in self._convert_to_series(pkg):
                yield token
            return
        # 3. 正向表达:继续/满意 → 推荐套餐
        if _is_convert_intent(user_input) or _is_positive_continue(s):
            async for token in self._recommend_packages_for_convert():
                yield token
            return
        yield "[REPLY][试讲顾问]没太明白,回复套餐序号/名称即可;或告诉我是否想继续上正式课。\n"

    async def _recommend_packages_for_convert(self) -> AsyncGenerator[str, None]:
        """基于试讲老师 + 学员画像推荐正式课套餐(Top-3)。

        复用 agents.trial.conversion_logic.ConversionLogic.recommend_packages
        (与 api/trial.py recommend-packages 同一 service 逻辑)。
        结果存入 trial_history["convert_packages"] 供下轮套餐选择匹配。
        """
        from agents.trial.conversion_logic import ConversionLogic

        teacher_id = self.trial_history.get("convert_teacher_id")
        lesson_id = self.trial_history.get("convert_lesson_id")
        try:
            student = self.processor.student_service.initialize_default_student()
        except Exception:
            student = None
        if not student:
            yield "[REPLY][试讲顾问]未找到学员信息,请稍后再试。\n"
            return
        # 用对话已确认的学科/年级覆盖默认画像(默认学员是高中小明,不匹配真实孩子)。
        # 学科/年级从试讲上下文拿,非空且非『未知』时覆盖,套餐打分自然跟随真实孩子。
        subject = self.trial_history.get("subject") or self.trial_history.get("convert_subject")
        grade_level = self.trial_history.get("grade_level")
        if subject and subject not in ("", "未知"):
            student["weak_subjects"] = [subject]
        if grade_level and grade_level not in ("", "未知"):
            student["grade_level"] = grade_level
        try:
            logic = ConversionLogic()
            packages = logic.recommend_packages(
                student=student, teacher_id=teacher_id,
                trial_lesson_id=lesson_id, limit=3,
            )
        except Exception as e:
            yield f"[REPLY][试讲顾问]推荐套餐时出错:{e}\n"
            return
        self.trial_history["convert_packages"] = packages
        if not packages:
            yield "[REPLY][试讲顾问]暂时没有匹配的正式课套餐,您可以稍后查看套餐页面。\n"
            return
        lines = ["[REPLY][试讲顾问]根据孩子情况,为您推荐以下正式课套餐:\n"]
        for idx, pkg in enumerate(packages, 1):
            ppp = pkg.get("price_per_lesson")
            per = f"{ppp:.0f}" if ppp is not None else "?"
            lines.append(
                f"{idx}. {pkg.get('name', '')}\n"
                f"   - {pkg.get('total_lessons', 0)} 课时,每节 {pkg.get('duration_minutes', 60)} 分钟\n"
                f"   - 总价 ¥{pkg.get('price_total', 0)}(单次 ¥{per})\n"
                f"   - 推荐: {pkg.get('match_reason', '')}\n"
            )
        lines.append("回复套餐序号或名称即可购买并转正式课。")
        yield "\n".join(lines)

    def _match_package_selection(self, s: str):
        """在已推荐套餐(convert_packages)里按序号/名称匹配家长的选择。

        支持:数字(1)、『套餐1』『套餐一』、『第1个』『第一个』、套餐名称子串。
        匹配失败返回 None(调用方继续按正向表达处理)。
        """
        from .trial_processor import _chinese_digit_to_int
        packages = self.trial_history.get("convert_packages") or []
        if not packages:
            return None
        import re as _re
        s2 = (s or "").strip()
        # 纯数字
        if s2.isdigit():
            idx = int(s2)
            if 1 <= idx <= len(packages):
                return packages[idx - 1]
        # 套餐X / 第X个
        m = _re.search(r"套餐\s*([0-9一二三四五六七八九十]+)", s2)
        if m:
            idx = _chinese_digit_to_int(m.group(1))
            if 1 <= idx <= len(packages):
                return packages[idx - 1]
        m = _re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*个", s2)
        if m:
            idx = _chinese_digit_to_int(m.group(1))
            if 1 <= idx <= len(packages):
                return packages[idx - 1]
        # 套餐名称子串匹配
        for pkg in packages:
            name = pkg.get("name") or ""
            if name and name in s2:
                return pkg
        return None

    async def _convert_to_series(self, pkg: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """把试讲转成正式系列课。

        复用 LessonService.create_lesson_series——它内部已含 ScheduleGenerator
        展开系列课 + add_series + 批量 add_lesson,与 api/trial.py convert 端点
        同一转换链路(5 层单向依赖:agent 层经 service 层落库,不直接 import api 层)。
        排课参数从套餐自带 weekly_schedule + 课时数推算(customer 未逐条给出每周时间)。
        """
        lesson_id = self.trial_history.get("convert_lesson_id")
        if not lesson_id:
            yield "[REPLY][试讲顾问]未找到对应试讲课时,请重新预约。\n"
            return
        from services.lesson_service import LessonService
        from config.time_config import time_config
        from agents.lesson import ScheduleGenerator

        try:
            student = self.processor.student_service.initialize_default_student()
        except Exception:
            student = None
        if not student or not student.get("id"):
            yield "[REPLY][试讲顾问]未找到学员信息,无法创建正式课。\n"
            return
        teacher_id = int(pkg.get("teacher_id") or self.trial_history.get("convert_teacher_id") or 0)
        package_id = int(pkg.get("id"))
        weekly = pkg.get("weekly_schedule") or [{"day_of_week": 2, "start_time": "19:00"}]
        total_lessons = int(pkg.get("total_lessons") or 12)
        per_week = max(1, len(weekly))
        total_weeks = max(1, (total_lessons + per_week - 1) // per_week)
        try:
            svc = LessonService()
            next_monday = ScheduleGenerator().find_next_monday(time_config.today().date())
            success, result = svc.create_lesson_series(
                teacher_id=teacher_id,
                student_id=student["id"],
                weekly_pattern=weekly,
                start_date=next_monday,
                total_weeks=total_weeks,
                subject=pkg.get("subject") or self.trial_history.get("convert_subject"),
                duration_minutes=int(pkg.get("duration_minutes") or 60),
                package_id=package_id,
                skip_conflicts=True,
            )
        except Exception as e:
            yield f"[REPLY][试讲顾问]转正式课失败:{e}\n"
            return
        self._clear_convert_state()
        if success and isinstance(result, dict):
            series_id = result.get("series_id")
            created = result.get("lessons_created", 0)
            yield (
                f"[REPLY][试讲顾问]🎉 已成功转为正式课!\n"
                f"套餐: {pkg.get('name', '')}\n"
                f"共创建 {created} 节课时(系列 #{series_id})。\n"
                f"已按每周固定时段为您排好课,可随时回复调整。\n"
            )
        else:
            err = result.get("error") if isinstance(result, dict) else str(result)
            yield f"[REPLY][试讲顾问]转正式课失败: {err}\n"

    def _clear_convert_state(self):
        """清空转化态相关字段(awaiting_convert 及 convert_* 辅助字段)。"""
        for key in ("awaiting_convert", "convert_teacher_id", "convert_subject",
                    "convert_lesson_id", "convert_packages"):
            self.trial_history.pop(key, None)

    async def _handle_time_followup(self, user_input) -> AsyncGenerator[str, None]:
        """家长在「请问具体几点」追问后补充具体时间的处理。

        确定性解析(不调 LLM):先拼 date_hint 再解析(如 "2026-09-25 晚上7点"),
        失败再直接解析原文自带的相对/绝对时间(如「周五晚上7点」)。成功则把绝对
        时间写回 trial_history["start_time"]、清追问标志,并直接走 handle_complete_trial
        继续建课(复用完整流程,不重跑 LLM 字段提取);失败则计数追问,最多 2 次,之后兜底。
        """
        from config.time_config import time_config

        hint = self.trial_history.get("pending_date_hint", "")
        user_in = (user_input or "").strip()

        dt = None
        # ① 拼 date_hint:家长只补了「几点」,日期由上一轮已解析出的上下文带出
        if hint:
            try:
                dt = time_config.parse_datetime(f"{hint} {user_in}")
            except Exception:
                dt = None
        # ② 失败直接解析原文(家长可能重说了完整时间,如「周五晚上7点」)
        if dt is None:
            try:
                dt = time_config.parse_datetime(user_in)
            except Exception:
                dt = None

        if dt is not None:
            self.trial_history["start_time"] = dt.strftime("%Y-%m-%d %H:%M")
            # 清追问标志与次数(防呆,避免下一轮再次走追问分支)
            for key in ("pending_time_query", "pending_date_hint", "pending_time_attempts"):
                self.trial_history.pop(key, None)
            # 直接继续建课(复用 handle_complete_trial 完整流程,不重跑 LLM)
            async for token in self.processor.handle_complete_trial(
                self.trial_history, self.session_id,
                chat_history=self.chat_history.messages,
            ):
                yield token
            # handle_complete_trial 走 direct trial 分支后会存 pending_trial,
            # 需把 state 切到 TRIAL_CONFIRM(与 _run_stream_impl 里同型),
            # 让主路由下一轮『确认』时跳过分类器直接转给本 Agent。
            if self.trial_history.get("pending_trial"):
                if self.state:
                    from agents.task_classification.state_manager import FlowState
                    self.state.value = FlowState.TRIAL_CONFIRM
            return

        # 解析失败:计数再追问,最多 2 次
        attempts = int(self.trial_history.get("pending_time_attempts", 0)) + 1
        self.trial_history["pending_time_attempts"] = attempts
        if attempts < 2:
            yield "[REPLY][试讲顾问]还是没识别到具体时间，例如回复：晚上7点\n"
        else:
            # 兜底:清掉追问标志,让家长重新完整描述时间(含日期)
            for key in ("pending_time_query", "pending_date_hint", "pending_time_attempts"):
                self.trial_history.pop(key, None)
            yield "[REPLY][试讲顾问]无法识别时间，请换个说法（如：周五晚上7点）\n"

    def _reset_state_after_trial(self):
        self.reset()
        if self.state:
            from config.constants import StateEnum
            self.state.value = StateEnum.CLASSIFY