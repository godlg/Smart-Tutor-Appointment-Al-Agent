"""
TrialAgent 时间解析失败时改为「追问具体时间」测试

背景:用户输入「这个礼拜五把」→ LLM 字段提取出 start_time="2026-09-25 未知"
(日期对、小时"未知")→ time_config.parse_datetime 返回 None → 旧代码直接回复
「无法识别时间...请换个说法」,流程中断。

修复(trial_time_followup spec):
1. trial_processor.handle_complete_trial 时间解析失败 / start_time 缺失时,
   不再 yield「无法识别」,而是 yield「请问具体几点」追问,并设
   trial_history["pending_time_query"] + pending_date_hint(带日期上下文)。
2. TrialAgent._run_stream_impl 入口加 pending_time_query 分支:家长补时间时
   走 _handle_time_followup 确定性解析(不调 LLM),成功写回 start_time 并继续建课;
   命中否定词(不约了/取消等)则清标志走正常流程。
3. _handle_time_followup 解析失败时计数追问,最多 2 次,仍失败才兜底。

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trial.trial_agent import TrialAgent  # noqa: E402
from agents.trial.trial_processor import (  # noqa: E402
    TrialProcessor,
    _extract_date_part,
    _is_time_negation,
)


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


# 本周 Friday(today 2026-09-19 是周六,故本周五 = 2026-09-25)
THIS_FRIDAY = "2026-09-25"

# 上游已确认老师(dict,TeacherService.get_teacher_by_id 返回)
TEACHER = {
    "id": 1, "name": "周敏", "gender": "女", "subjects": ["数学"],
    "grade_levels": ["初中"], "teaching_style": "温和", "hourly_rate": 200,
    "years_experience": 5, "highlights": ["耐心"], "bio": "资深数学老师",
}


# ===========================================================================
# 辅助工具
# ===========================================================================

class _Parser:
    """假 TrialInputParser:返回可控的字段提取 JSON。

    parser_start_time 控制本轮 LLM「提取」出的 start_time(喂给状态机)。
    追问后的轮次(_handle_time_followup)不走 parser,所以默认值无所谓。
    """

    def __init__(self, parser_start_time: str):
        self.parser_start_time = parser_start_time

    def parse_stream(self, user_input, chat_history, context=""):
        yield (
            '{'
            '"subject":"数学","grade_level":"初中",'
            f'"start_time":"{self.parser_start_time}","info_complete":true'
            '}'
        )

    def parse_data(self, _s):
        return {
            "subject": "数学", "grade_level": "初中",
            "start_time": self.parser_start_time, "duration_minutes": "30",
            "student_situation": "", "preferred_teacher_name": "周敏",
            "preferred_style": "未知", "confirmation": "未知",
            "info_complete": True, "unrelated": False, "missing_info": [],
        }


def _make_agent(pending_time: bool = False, date_hint: str = "",
                parser_start_time: str = "未知") -> TrialAgent:
    """绕过 __init__(避免真实 LLM),构造 TrialAgent-like 实例。

    可手动塞 pending_time_query + 测 _handle_time_followup 路径与
    handle_complete_trial 的时间解析分支。
    """
    agent = object.__new__(TrialAgent)
    agent.session_id = "test-time-session"

    class _StubProcessor:
        pass

    real_tp = _StubProcessor()
    real_tp.input_parser = _Parser(parser_start_time)
    real_tp.teacher_finder = MagicMock()
    real_tp.teacher_finder.find_recommended_teachers.return_value = []
    real_tp.teacher_finder.check_teacher_availability.return_value = True
    real_tp.lesson_service = MagicMock()
    real_tp.lesson_service.create_lesson.return_value = (True, 9999)
    real_tp.student_service = MagicMock()
    real_tp.student_service.initialize_default_student.return_value = {"id": 1}
    real_tp.message_builder = MagicMock()
    real_tp.message_builder.create_direct_trial_message.return_value = (
        "\n顾问:✅ 已为您预约 周敏 老师...试讲。确认请回复『确认』\n"
    )
    real_tp.message_builder.create_trial_success_message.return_value = "试讲成功"
    real_tp.message_builder.create_recommendation_message.return_value = "推荐"
    real_tp.message_builder.create_missing_info_questions.return_value = "missing"
    real_tp.message_builder.create_parse_error_message.return_value = "parse err"
    real_tp.message_builder.create_unrelated_message.return_value = "unrelated"
    real_tp.handle_complete_trial = TrialProcessor.handle_complete_trial.__get__(real_tp)
    real_tp._create_trial = TrialProcessor._create_trial.__get__(real_tp)
    real_tp.update_history_from_data = TrialProcessor.update_history_from_data.__get__(real_tp)
    real_tp.handle_incomplete_info = TrialProcessor.handle_incomplete_info.__get__(real_tp)
    real_tp.handle_unrelated_request = TrialProcessor.handle_unrelated_request.__get__(real_tp)

    agent.processor = real_tp

    agent.trial_history = {
        "subject": "数学", "grade_level": "初中",
        "start_time": None, "duration_minutes": "30",
        "student_situation": None,
        "preferred_teacher_name": "周敏", "preferred_style": None,
        "teacher_id": TEACHER["id"],
        "unrelated_escalations": 0,
    }
    if pending_time:
        agent.trial_history["pending_time_query"] = True
        agent.trial_history["pending_date_hint"] = date_hint

    agent.chat_history = MagicMock()
    agent.chat_history.messages = []
    agent.chat_history.add_message = lambda m: None
    agent.chat_history.clear = lambda: None

    agent.state = MagicMock()  # 允许 .value 任意赋值
    agent.state.value = "classify"

    agent.unrelated_callback = None
    agent.context_manager = None
    agent.session_chat_history = None
    agent.finished = False
    agent._record_shared_ai = lambda c: None
    return agent


# ===========================================================================
# 用例 A:输入「这个礼拜五把」→ 输出「几点」追问,设 pending_time_query +
# pending_date_hint == 本周五日期
# ===========================================================================

class TestParseFailYieldsFollowup:
    def test_sets_pending_and_asks_time(self):
        # 模拟 LLM 提取出残缺时间(日期对、小时"未知")
        agent = _make_agent(parser_start_time=f"{THIS_FRIDAY} 未知")

        out = _collect(agent.run_stream(user_input="这个礼拜五把"))

        # 输出含「几点」追问,且带日期上下文;不含旧文案「无法识别...请换个说法」
        assert "几点" in out
        assert THIS_FRIDAY in out
        assert "无法识别" not in out
        # 状态:设追问标志 + 日期 hint
        assert agent.trial_history.get("pending_time_query") is True
        assert agent.trial_history.get("pending_date_hint") == THIS_FRIDAY

    def test_missing_start_time_sets_pending(self):
        # start_time 缺失分支在 handle_complete_trial 内部,需学科/年级已齐、
        # 仅 start_time 为「未知」时触发(通常走完缺信息追问后进入)。
        # 这里直接调 handle_complete_trial 验证该分支。
        agent = _make_agent()
        agent.trial_history["start_time"] = "未知"

        out = _collect(agent.processor.handle_complete_trial(
            agent.trial_history, agent.session_id,
            chat_history=agent.chat_history.messages,
        ))

        assert "几点" in out
        assert "晚上7点" in out
        # 不带日期上下文(完全没有可提取的日期)
        assert agent.trial_history.get("pending_time_query") is True
        assert agent.trial_history.get("pending_date_hint") == ""


# ===========================================================================
# 用例 B:追问后输入「晚上7点」→ 解析成功 19:00,继续建课(不出现「无法识别」)
# ===========================================================================

class TestFollowupParsesTime:
    def test_evening_7pm(self):
        agent = _make_agent(pending_time=True, date_hint=THIS_FRIDAY)

        with patch("services.teacher_service.TeacherService") as TS:
            TS.return_value.get_teacher_by_id.return_value = TEACHER
            TS.return_value.get_teacher_by_name.return_value = None
            out = _collect(agent.run_stream(user_input="晚上7点"))

        # 不出现旧错误文案
        assert "无法识别" not in out
        assert "请换个说法" not in out
        # start_time 解析成功(拼日期 hint → 本周五 19:00)
        assert agent.trial_history.get("start_time") == f"{THIS_FRIDAY} 19:00"
        # 追问标志已清
        assert agent.trial_history.get("pending_time_query") is None
        # 继续走建课流程:yield direct trial 引导消息 + 落 pending_trial
        assert "已为您预约" in out
        assert agent.trial_history.get("pending_trial") is not None

    def test_afternoon_3pm(self):
        # 用例 C:下午3点 → 15:00
        agent = _make_agent(pending_time=True, date_hint=THIS_FRIDAY)

        with patch("services.teacher_service.TeacherService") as TS:
            TS.return_value.get_teacher_by_id.return_value = TEACHER
            TS.return_value.get_teacher_by_name.return_value = None
            out = _collect(agent.run_stream(user_input="下午3点"))

        assert "无法识别" not in out
        assert agent.trial_history.get("start_time") == f"{THIS_FRIDAY} 15:00"
        assert "已为您预约" in out


# ===========================================================================
# 用例 D:连续两次乱输入 → 第一次再追问、第二次兜底「请换个说法」+ 清标志
# ===========================================================================

class TestFollowupRetryThenFallback:
    def test_two_failures_then_fallback(self):
        # 第一次乱输入「随便」→ 计数 1 < 2,继续追问
        agent = _make_agent(pending_time=True, date_hint=THIS_FRIDAY)
        out1 = _collect(agent.run_stream(user_input="随便"))
        assert "还是没识别到具体时间" in out1
        assert agent.trial_history.get("pending_time_query") is True
        assert agent.trial_history.get("pending_time_attempts") == 1

        # 第二次乱输入「都行」→ 计数 2,兜底「请换个说法」+ 清标志
        out2 = _collect(agent.run_stream(user_input="都行"))
        assert "请换个说法" in out2
        # 标志清空,不再进入追问分支
        assert agent.trial_history.get("pending_time_query") is None
        assert agent.trial_history.get("pending_time_attempts") is None
        assert agent.trial_history.get("pending_date_hint") is None

    def test_three_failures_third_goes_normal(self):
        # 兜底清标志后,第三次输入进入正常解析(parser 路径),不再抛异常
        agent = _make_agent(pending_time=True, date_hint=THIS_FRIDAY)
        _collect(agent.run_stream(user_input="随便"))
        out2 = _collect(agent.run_stream(user_input="都行"))
        assert "请换个说法" in out2
        out3 = _collect(agent.run_stream(user_input="再瞎说"))
        # 不再重复追问(标志已清),走正常流程(parser 返回残缺信息 → 缺信息追问)
        assert agent.trial_history.get("pending_time_query") is None
        assert out3  # 有输出,不抛异常


# ===========================================================================
# 用例 E:追问后输入「不约了」→ 清标志走正常流程,不当作时间解析
# ===========================================================================

class TestNegationClearsPending:
    def test_cancel_clears_flags(self):
        agent = _make_agent(pending_time=True, date_hint=THIS_FRIDAY)

        out = _collect(agent.run_stream(user_input="不约了"))

        # 否定词 → 清追问标志,不把「不约了」当时间解析
        assert agent.trial_history.get("pending_time_query") is None
        assert agent.trial_history.get("pending_date_hint") is None
        # 不应进入「还想我补充时间」的兜底回复
        assert "无法识别时间，请换个说法" not in out
        # 走正常流程(parser 返回残缺信息 → 缺信息追问)
        assert "信息不完整" in out

    def test_is_time_negation_keywords(self):
        # 纯关键字判定
        for s in ("不约了", "取消吧", "算了", "不要了", "换老师吧", "换时间"):
            assert _is_time_negation(s), f"应判 True: {s!r}"
        for s in ("晚上7点", "下午3点", "周五晚上", "随便"):
            assert not _is_time_negation(s), f"应判 False: {s!r}"
        assert not _is_time_negation("")
        assert not _is_time_negation(None)


# ===========================================================================
# 辅助函数单测
# ===========================================================================

class TestExtractDatePart:
    def test_extracts_date(self):
        assert _extract_date_part("2026-09-25 未知") == "2026-09-25"
        assert _extract_date_part("2026-09-25") == "2026-09-25"
        assert _extract_date_part("未知") == ""
        assert _extract_date_part("") == ""
        assert _extract_date_part(None) == ""