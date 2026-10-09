"""
TrialAgent 在『pending_trial』上下文里处理家长『确认』直接创建 Lesson(TRIAL) 测试

背景:TrialAgent 上一轮 yield 了『已为您预约 张明 老师...试讲。确认请回复『确认』』
的引导消息后,家长本轮回『确认』,TaskClassifier LLM 会把它误判成 recommendation
→ 走 RecommendationAgent → 追问学科年级(典型 Round 5 bug)。

修复(trial_confirm_route spec):
1. TrialProcessor.handle_complete_trial yield direct trial 引导消息时存
   trial_history["pending_trial"] = {teacher_id, teacher_name, subject, grade_level,
   time_str, time_iso, duration_minutes, created_at}
2. TrialProcessor 加 _is_simple_confirm(user_input) 关键字判定
3. TrialAgent._run_stream_impl 入口加分支 0:有 pending_trial + 简单确认 →
   调 _confirm_pending_trial 真正创建 Lesson(TRIAL),不调 LLM、不走 TrialInputParser
4. StateManager 加 TRIAL_CONFIRM 状态 + transition_to_trial_confirm
5. AgentRouter.route_to_trial_confirm 不 yield 归类机器人首轮分流文案
6. ClassificationProcessor 在 TRIAL_CONFIRM 状态时跳过分类器,直接转给 TrialAgent

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.task_classification.state_manager import (  # noqa: E402
    StateManager, FlowState,
)
from agents.task_classification.agent_router import AgentRouter  # noqa: E402
from agents.task_classification.classification_processor import (  # noqa: E402
    ClassificationProcessor,
)
from agents.task_classification.unrelated_handler import UnrelatedHandler  # noqa: E402
from agents.trial.trial_agent import TrialAgent  # noqa: E402
from agents.trial.trial_processor import (  # noqa: E402
    TrialProcessor,
    _is_simple_confirm,
    _create_pending_trial_lesson,
    extract_confirmed_teacher_id,
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


# ===========================================================================
# Case 5(优先):_is_simple_confirm 单元测试
# ===========================================================================

class TestIsSimpleConfirm:
    """_is_simple_confirm 关键字判定,纯字符串处理,无 LLM"""

    # 正向:常见中文肯定回复
    def test_positive_chinese_basic(self):
        for s in ("是", "好", "可以", "行", "确认", "好呀", "好的", "行吧", "可以呀", "可以啊"):
            assert _is_simple_confirm(s), f"应判 True: {s!r}"

    # 正向:常见英文肯定回复
    def test_positive_english_basic(self):
        for s in ("ok", "OK", "yes", "Yes", "YES", "y", "Y", "okay", "OKAY"):
            assert _is_simple_confirm(s), f"应判 True: {s!r}"

    # 正向:扩展肯定(如『好的呢』『可以的』『没问题』)
    def test_positive_extended(self):
        for s in ("好的呢", "好的吧", "是的", "对的", "没问题", "可以的", "可以吧", "ok啦", "ok的"):
            assert _is_simple_confirm(s), f"应判 True: {s!r}"

    # 正向:『确认…』起始词(『确认时间』『确认安排』『确认了吗?』—— 最后一个含问号,负向)
    def test_positive_prefix(self):
        assert _is_simple_confirm("确认")
        assert _is_simple_confirm("确认时间")
        assert _is_simple_confirm("确认安排")
        assert _is_simple_confirm("确认预约")
        assert _is_simple_confirm("确认好啦")

    # 负向:疑问句结尾
    def test_negative_questions(self):
        assert not _is_simple_confirm("确认吗?"), "应判 False"
        assert not _is_simple_confirm("是吗?")
        assert not _is_simple_confirm("要确认吗?")
        assert not _is_simple_confirm("确认吗？")  # 全角问号

    # 负向:否定前缀
    def test_negative_negation(self):
        assert not _is_simple_confirm("不是")
        assert not _is_simple_confirm("不要")
        assert not _is_simple_confirm("不要确认")
        assert not _is_simple_confirm("不是确认")
        assert not _is_simple_confirm("先不")

    # 负向:修改时间/选老师等明确意图(应走原 TrialInputParser)
    def test_negative_modification(self):
        assert not _is_simple_confirm("改成周三晚上 8 点")
        assert not _is_simple_confirm("改成明天")
        assert not _is_simple_confirm("张明")
        assert not _is_simple_confirm("周敏老师")
        assert not _is_simple_confirm("算了吧")

    # 负向:空输入/None
    def test_negative_empty(self):
        assert not _is_simple_confirm("")
        assert not _is_simple_confirm(None)
        assert not _is_simple_confirm("   ")  # 空白

    # 边界:带前后空白
    def test_trim_whitespace(self):
        assert _is_simple_confirm("  确认  ")
        assert _is_simple_confirm(" 是\n")


# ===========================================================================
# Case 3:无 pending_trial + 输入『确认』→ fallback(走原分支)
# ===========================================================================

class TestNoPendingTrialFallsBack:
    """无 pending_trial 时,TrialAgent 不走 confirm 分支,继续走 TrialInputParser"""

    def test_no_pending_trial_normalizes_through_parser(self):
        agent = _make_trial_agent_with_pending(pending=None)
        # input_parser 故意返回未知(info_complete=False) → 触发 handle_incomplete_info
        agent.processor.input_parser.parse_data = lambda s: {
            "subject": "未知", "grade_level": "未知",
            "start_time": "未知", "duration_minutes": "30",
            "student_situation": "", "preferred_teacher_name": "未知",
            "preferred_style": "未知", "confirmation": "未知",
            "info_complete": False, "unrelated": False,
            "missing_info": ["subject", "grade_level"],
        }

        def _stream_unknown(*a, **kw):
            yield '{"subject":"未知","grade_level":"未知","info_complete":false}'
        agent.processor.input_parser.parse_stream = _stream_unknown

        out = _collect(agent.run_stream(user_input="确认"))

        # ❌ 不应 yield 试讲已成功预约(没有 pending_trial 就不走 confirm 分支)
        assert "试讲已成功预约" not in out
        # 走 handle_incomplete_info → yield missing_info 追问
        # 注意:具体追问文案由 message_builder 决定,只要不含『成功预约』即可
        assert "成功预约" not in out

    def test_no_pending_trial_pending_value_is_none(self):
        """helper 在 pending=None 时 trial_history.get("pending_trial") 应返回 None。"""
        agent = _make_trial_agent_with_pending(pending=None)
        # 关键:值是 None(由 helper 控制),不要混进 mock DB / 走 confirm 分支
        assert agent.trial_history.get("pending_trial") is None


# ===========================================================================
# Case 1 + Case 2:有 pending_trial + 输入『确认』/『是』→ 创建 Lesson
# ===========================================================================

class TestPendingTrialConfirmCreatesLesson:
    """TrialAgent 入口见到 pending_trial + 简单确认 → 调 _confirm_pending_trial
    真正创建 Lesson(TRIAL),yield 成功消息,清 pending。"""

    def test_case1_confirm_creates_lesson(self):
        """Case 1:『确认』 → 创建 Lesson + yield 成功消息 + 清 pending"""
        pending = _make_pending_trial()
        agent = _make_trial_agent_with_pending(pending=pending)

        out = _collect(agent.run_stream(user_input="确认"))

        # ✅ yield 成功预约消息(3 个关键字任选其一即可)
        assert (
            "🎉 试讲已成功预约" in out
            or "课程 ID" in out
            or "Lesson" in out
        ), f"未 yield 成功消息: {out!r}"

        # ✅ lesson_service.create_lesson 被调(真实落库)
        agent.processor.lesson_service.create_lesson.assert_called_once()
        call_kwargs = agent.processor.lesson_service.create_lesson.call_args.kwargs
        # 创建参数:lesson_type='trial',status='confirmed',price=0.0
        assert call_kwargs.get("lesson_type") == "trial"
        assert call_kwargs.get("status") == "confirmed"
        assert call_kwargs.get("price") == 0.0
        assert call_kwargs.get("teacher_id") == pending["teacher_id"]
        assert call_kwargs.get("duration_minutes") == pending["duration_minutes"]

        # ✅ pending_trial 清空
        assert "pending_trial" not in agent.trial_history
        # trial_history 也被 reset(回到 initial 状态)
        assert agent.trial_history["subject"] is None

    def test_case2_shi_creates_lesson(self):
        """Case 2:『是』 → 同样创建 Lesson(TRIAL)"""
        pending = _make_pending_trial()
        agent = _make_trial_agent_with_pending(pending=pending)

        out = _collect(agent.run_stream(user_input="是"))

        assert "🎉 试讲已成功预约" in out
        agent.processor.lesson_service.create_lesson.assert_called_once()
        assert "pending_trial" not in agent.trial_history

    def test_case2_yes_creates_lesson(self):
        """Case 2 扩展:英文 yes 也走 confirm 分支"""
        pending = _make_pending_trial()
        agent = _make_trial_agent_with_pending(pending=pending)

        _collect(agent.run_stream(user_input="yes"))
        agent.processor.lesson_service.create_lesson.assert_called_once()

    def test_case1_confirm_includes_teacher_and_time(self):
        """成功消息必须带老师姓名 + 时间(从 pending 拷过去)"""
        pending = _make_pending_trial()
        pending["teacher_name"] = "张明"
        pending["time_str"] = "2026-09-16 19:00"
        agent = _make_trial_agent_with_pending(pending=pending)

        out = _collect(agent.run_stream(user_input="确认"))
        assert "张明" in out
        assert "2026-09-16 19:00" in out

    def test_pending_cleared_after_confirm(self):
        """确认成功后 pending_trial 必须清掉,避免下一轮再次走 confirm 分支。

        验证方式:首次 confirm 后 trial_history 不含 pending_trial。
        (额外做:再次『确认』输入不再走 confirm 分支,因为 trial_history 已 reset
        + 没有 pending_trial 锚点;此时会走 TrialInputParser 解析,
        与『正常家长在 TrialAgent 上下文』行为一致,不再触发创建 Lesson 的副作用)
        """
        pending = _make_pending_trial()
        agent = _make_trial_agent_with_pending(pending=pending)

        _collect(agent.run_stream(user_input="确认"))
        # 第一轮后 pending_trial 必须被清(防呆)
        assert "pending_trial" not in agent.trial_history
        # trial_history 也被 reset(回 CLASSIFY 兜底态)
        assert agent.trial_history["subject"] is None


# ===========================================================================
# Case 4:有 pending_trial + 输入『改成周三晚上 8 点』→ 不走 confirm,走原 parser
# ===========================================================================

class TestModificationIntentFallsThrough:
    """修改时间/选老师等明确意图不触发 confirm 分支,走 TrialInputParser"""

    def test_case4_modify_time_falls_through(self):
        pending = _make_pending_trial()
        agent = _make_trial_agent_with_pending(pending=pending)
        # input_parser 模拟返回『修改时间』意图(info_complete=False,subject/grade 已存在)
        agent.processor.input_parser.parse_data = lambda s: {
            "subject": "数学", "grade_level": "初中",
            "start_time": "2026-09-17 20:00",  # 改到周三晚 8 点
            "duration_minutes": "30", "student_situation": "",
            "preferred_teacher_name": "张明", "preferred_style": "未知",
            "confirmation": "未知", "info_complete": True,
            "unrelated": False, "missing_info": [],
        }

        def _stream_change(*a, **kw):
            yield '{"subject":"数学","grade_level":"初中","start_time":"2026-09-17 20:00","info_complete":true}'
        agent.processor.input_parser.parse_stream = _stream_change

        prior_calls = agent.processor.lesson_service.create_lesson.call_count
        out = _collect(agent.run_stream(user_input="改成周三晚上 8 点"))

        # ❌ 不应调 create_lesson(『改成 X』不是简单确认,不该落库)
        # 实际上此时 trial_processor 走的是『列 3 个老师再 confirm』还是
        # 『direct trial 引导消息』取决于 trial_history 已有 teacher_id 时
        # 走 direct trial 分支再 yield 一遍『确认请回复』。两种情况都不该
        # 立即 create_lesson。
        assert agent.processor.lesson_service.create_lesson.call_count == prior_calls
        # ❌ 不应 yield 成功预约消息(用户还没确认)
        assert "🎉 试讲已成功预约" not in out


# ===========================================================================
# 额外:StateManager / AgentRouter / ClassificationProcessor
# ===========================================================================

class TestStateManagerTrialConfirm:
    """StateManager 支持 TRIAL_CONFIRM 状态转换与判定"""

    def test_default_is_classify(self):
        sm = StateManager()
        assert sm.get_current_state() == FlowState.CLASSIFY
        assert sm.is_in_trial_confirm_flow() is False

    def test_transition_to_trial_confirm(self):
        sm = StateManager()
        sm.transition_to_trial_confirm()
        assert sm.get_current_state() == FlowState.TRIAL_CONFIRM
        assert sm.is_in_trial_confirm_flow() is True
        # transition 后 should_classify=False(下一轮不再走分类器)
        assert sm.should_classify() is False

    def test_state_description_includes_trial_confirm(self):
        sm = StateManager()
        sm.transition_to_trial_confirm()
        assert sm.get_state_description() == "试讲预约确认态"


class TestRouteToTrialConfirm:
    """route_to_trial_confirm 不 yield 归类机器人首轮分流文案"""

    def test_route_skips_classification_thought(self):
        agent = _FakeTrialAgent()
        router = AgentRouter(
            trial_agent=agent, state_manager=MagicMock(),
        )
        out = _run(_drain(router.route_to_trial_confirm("确认")))
        # 关键:不应出现「[THOUGHT][归类机器人] 这是一个试讲预约请求」
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" not in out
        assert "试讲确认成功" in out
        assert agent.calls == ["确认"]

    def test_route_without_agent_yields_error(self):
        router = AgentRouter(trial_agent=None, state_manager=MagicMock())
        out = _run(_drain(router.route_to_trial_confirm("任意输入")))
        assert "[ERROR]" in out

    def test_route_differs_from_route_to_trial(self):
        """与 route_to_trial 对比:后者有分流文案,前者没有"""
        agent = _FakeTrialAgent()
        router = AgentRouter(trial_agent=agent, state_manager=MagicMock())
        out1 = _run(_drain(router.route_to_trial("任意输入")))
        out2 = _run(_drain(router.route_to_trial_confirm("任意输入")))
        # 老路由有分流文案
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" in out1
        # 新路由不带
        assert "[THOUGHT][归类机器人]" not in out2


class TestClassificationProcessorTrialConfirmBypass:
    """主路由在 TRIAL_CONFIRM 状态时跳过分类器,直接转给 TrialAgent"""

    def test_stream_skips_classifier_in_trial_confirm_state(self):
        sm = StateManager()
        sm.transition_to_trial_confirm()
        trial_agent = _FakeTrialAgent()
        router = AgentRouter(
            consultant_agent=None, state_manager=sm,
            trial_agent=trial_agent, lesson_agent=None,
            recommendation_agent=None,
        )
        classifier = _CounterClassifier()
        processor = ClassificationProcessor(
            task_classifier=classifier, state_manager=sm,
            agent_router=router, unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        prior_calls = classifier.call_count
        out = _collect(processor.process_task_stream("确认"))
        # 关键:分类器没被调
        assert classifier.call_count == prior_calls
        # 输出不含归类机器人首轮分流文案
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" not in out

    def test_sync_skips_classifier_in_trial_confirm_state(self):
        sm = StateManager()
        sm.transition_to_trial_confirm()
        trial_agent = _FakeTrialAgent()
        router = AgentRouter(
            consultant_agent=None, state_manager=sm,
            trial_agent=trial_agent, lesson_agent=None,
            recommendation_agent=None,
        )
        classifier = _CounterClassifier()
        processor = ClassificationProcessor(
            task_classifier=classifier, state_manager=sm,
            agent_router=router, unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        prior_calls = classifier.call_count
        result = _run(processor.process_task_sync("是"))
        assert classifier.call_count == prior_calls
        assert "[THOUGHT][归类机器人]" not in result

    def test_recommendation_confirm_state_does_not_trigger_trial_bypass(self):
        """只判 TRIAL_CONFIRM,不串扰到 RECOMMENDATION_CONFIRM 分支"""
        sm = StateManager()
        sm.transition_to_recommendation_confirm()
        # trial_agent 没设,不应该走 trial_confirm 分支(只走 recommendation_confirm)
        router = AgentRouter(
            consultant_agent=None, state_manager=sm,
            trial_agent=None, lesson_agent=None,
            recommendation_agent=_FakeRecommendationAgent(),
        )
        classifier = _CounterClassifier()
        processor = ClassificationProcessor(
            task_classifier=classifier, state_manager=sm,
            agent_router=router, unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        prior_calls = classifier.call_count
        _collect(processor.process_task_stream("张明"))
        # 分类器没被调(走的是 recommendation_confirm 分支)
        assert classifier.call_count == prior_calls


# ===========================================================================
# 辅助工具
# ===========================================================================

class _FakeTrialAgent:
    """假 TrialAgent:记录被续接的输入,并给出回复"""

    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][试讲顾问]试讲确认成功"


class _FakeRecommendationAgent:
    """假 RecommendationAgent:记录被续接的输入,并给出回复"""

    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][推荐助手]推荐确认续轮结果"


class _CounterClassifier:
    """记录调用次数的分类器"""

    def __init__(self, return_category="recommendation_confirm"):
        self.call_count = 0
        self.return_category = return_category

    async def classify_task(self, task):
        self.call_count += 1
        return self.return_category


def _make_pending_trial() -> dict:
    """构造一个标准的 pending_trial dict(模拟 handle_complete_trial 落库结果)"""
    return {
        "teacher_id": 1,
        "teacher_name": "张明",
        "subject": "数学",
        "grade_level": "初中",
        "time_str": "2026-09-16 19:00",
        "time_iso": "2026-09-16T19:00:00",
        "duration_minutes": 30,
        "created_at": _time.time(),
    }


def _make_trial_agent_with_pending(pending: dict = None) -> TrialAgent:
    """绕过 __init__(避免真实 LLM),构造一个 TrialAgent-like 实例,
    可手动塞 pending_trial + 测 _confirm_pending_trial 路径。

    主要 mock:
    - input_parser.parse_stream:返回一个固定的 JSON
    - input_parser.parse_data:同上
    - processor.lesson_service.create_lesson:返回成功 + lesson_id=9999
    - processor.student_service.initialize_default_student:返回默认学员
    - processor.teacher_finder.find_recommended_teachers:返回空(防御)
    - state:Mock(允许 .value 赋值)
    """
    agent = object.__new__(TrialAgent)
    agent.session_id = "test-session"

    # processor:真实方法 + mock 依赖
    class _FakeParser:
        def parse_stream(self, user_input, chat_history, context=""):
            yield '{"subject":"数学","grade_level":"初中","start_time":"2026-09-16 19:00","info_complete":true}'

        def parse_data(self, _s):
            return {
                "subject": "数学", "grade_level": "初中",
                "start_time": "2026-09-16 19:00", "duration_minutes": "30",
                "student_situation": "", "preferred_teacher_name": "张明",
                "preferred_style": "未知", "confirmation": "未知",
                "info_complete": True, "unrelated": False, "missing_info": [],
            }

    class _StubProcessor:
        pass

    real_tp = _StubProcessor()
    real_tp.input_parser = _FakeParser()
    real_tp.teacher_finder = MagicMock()
    real_tp.teacher_finder.find_recommended_teachers.return_value = []
    real_tp.teacher_finder.check_teacher_availability.return_value = True
    real_tp.lesson_service = MagicMock()
    real_tp.lesson_service.create_lesson.return_value = (True, 9999)
    real_tp.student_service = MagicMock()
    real_tp.student_service.initialize_default_student.return_value = {"id": 1}
    real_tp.message_builder = MagicMock()
    real_tp.message_builder.create_direct_trial_message.return_value = (
        "\n顾问:✅ 已为您预约 张明 老师...试讲。确认请回复『确认』\n"
    )
    real_tp.message_builder.create_trial_success_message.return_value = "试讲成功"
    real_tp.message_builder.create_recommendation_message.return_value = "推荐"
    real_tp.message_builder.create_missing_info_questions.return_value = "missing"
    real_tp.message_builder.create_parse_error_message.return_value = "parse err"
    real_tp.message_builder.create_unrelated_message.return_value = "unrelated"
    # 绑真实方法
    real_tp.handle_complete_trial = TrialProcessor.handle_complete_trial.__get__(real_tp)
    real_tp._create_trial = TrialProcessor._create_trial.__get__(real_tp)
    real_tp.update_history_from_data = TrialProcessor.update_history_from_data.__get__(real_tp)
    real_tp.handle_incomplete_info = TrialProcessor.handle_incomplete_info.__get__(real_tp)
    real_tp.handle_unrelated_request = TrialProcessor.handle_unrelated_request.__get__(real_tp)

    agent.processor = real_tp

    # trial_history(可挂 pending_trial)
    agent.trial_history = {
        "subject": "数学", "grade_level": "初中",
        "start_time": "2026-09-16 19:00", "duration_minutes": "30",
        "student_situation": None,
        "preferred_teacher_name": "张明", "preferred_style": None,
        "teacher_id": 1,
        "unrelated_escalations": 0,
        "pending_trial": pending,
    }
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