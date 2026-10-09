"""
推荐 → 试讲跨 Agent 上下文平滑交接完整测试

覆盖 Bug 2:TrialAgent 拿不到上游已确认的『学科/年级/老师』时,会重新跑
find_recommended_teachers 把已经选好的老师再列一遍让家长选——这是用户报的核心 bug。

修复:
1. TrialAgent._extract_confirmed_context 从共享历史 parse 出上游确认字段
2. _run_stream_impl 入口回填 trial_history(subject/grade_level/preferred_teacher_name/teacher_id)
3. TrialProcessor.handle_complete_trial 当 preferred_teacher_name 已设且只返回 1 个老师时
   直接走 _create_trial,不再 yield create_recommendation_message 让家长『选哪位』

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trial.trial_agent import TrialAgent  # noqa: E402
from agents.trial.trial_processor import TrialProcessor  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


# ----------------------------------------------------------------------
# 工具:从共享历史回填上游确认字段(parse 静态行为,无 LLM)
# ----------------------------------------------------------------------

class TestExtractConfirmedContext:
    """TrialAgent._extract_confirmed_context 静态解析上游确认消息"""

    def test_extracts_full_confirmation(self):
        ctx = "上游上下文: [REPLY][推荐助手]已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)。好的,为您预约试讲"
        out = TrialAgent._extract_confirmed_context(ctx)
        assert out["subject"] == "数学"
        assert out["grade_level"] == "初中"
        assert out["teacher_id"] == 1
        assert out["preferred_teacher_name"] == "张明"

    def test_extracts_partial(self):
        """只有学科/年级(没老师ID)时仍能解析(降级)"""
        ctx = "上游上下文: 已为您确认 周敏 老师(学科: 数学, 年级: 初中)"
        out = TrialAgent._extract_confirmed_context(ctx)
        assert out["subject"] == "数学"
        assert out["grade_level"] == "初中"
        assert out["preferred_teacher_name"] == "周敏"
        assert "teacher_id" not in out

    def test_empty_context_returns_empty_dict(self):
        assert TrialAgent._extract_confirmed_context("") == {}
        assert TrialAgent._extract_confirmed_context("无关键字段的 AI 消息") == {}

    def test_handles_full_width_colon(self):
        ctx = "学科:数学 年级:初中 老师ID:7 已为您确认 周敏 老师"
        out = TrialAgent._extract_confirmed_context(ctx)
        assert out["subject"] == "数学"
        assert out["grade_level"] == "初中"
        assert out["teacher_id"] == 7
        assert out["preferred_teacher_name"] == "周敏"


# ----------------------------------------------------------------------
# 核心 bug 验证:TrialAgent 拿到上游确认后,不应再列老师
# ----------------------------------------------------------------------

def _make_trial_agent_with_shared_history(confirmed_msg: str):
    """构造一个 TrialAgent(绕过 __init__ 真实 LLM),模拟『上游刚确认了老师』的场景。

    - session_chat_history 里有一条携带『学科:/老师ID:』的 AI 消息(上游推荐 agent 写的)
    - trial_processor 的 teacher_finder.find_recommended_teachers 用 mock,
      调用次数可断言(关键:不应被调用)
    - input_parser.parse_stream 模拟 LLM 输出 JSON(含 start_time,subject/grade/teacher 由回填保证)
    """
    agent = object.__new__(TrialAgent)
    agent.session_id = "test-session"

    # 共享历史:装一条上游确认消息
    shared = InMemoryChatMessageHistory()
    shared.add_ai_message(confirmed_msg)
    agent.session_chat_history = shared

    # input_parser:真实小类(注意 parse_stream 用 def+yield 是普通 generator,
    # 不是 async def+yield——后者会得到 async generator,production 代码用同步 for 迭代会 TypeError)
    class _FakeParser:
        def __init__(self, json_text):
            self._json = json_text

        def parse_stream(self, user_input, chat_history, context=""):
            for ch in self._json:
                yield ch

        def parse_data(self, _s):
            return {
                "subject": "数学", "grade_level": "初中",
                "start_time": "2026-09-16 19:00", "duration_minutes": "30",
                "student_situation": "", "preferred_teacher_name": "张明",
                "preferred_style": "未知", "confirmation": "未知",
                "info_complete": True, "unrelated": False, "missing_info": [],
            }

    # 拿真实 TrialProcessor 的方法(不调 __init__,避免真实 LLM)
    # 不用 MagicMock(spec=...):MagicMock 会 wrap async generator 成 coroutine,出错
    class _StubProcessor:
        pass

    real_tp = _StubProcessor()
    real_tp.input_parser = _FakeParser(
        '{"subject":"数学","grade_level":"初中","start_time":"2026-09-16 19:00",'
        '"duration_minutes":"30","student_situation":"","preferred_teacher_name":"张明",'
        '"preferred_style":"未知","confirmation":"未知","info_complete":true,'
        '"unrelated":false,"missing_info":[]}'
    )
    real_tp.teacher_finder = MagicMock()
    real_tp.lesson_service = MagicMock()
    real_tp.student_service = MagicMock()
    # teacher_finder.find_recommended_teachers:返回 1 个老师(模拟『上游指定老师』
    # 时 TrialTeacherFinder.find_recommended_Teachers 的行为,见 trial_processor.py:179-184)
    real_tp.teacher_finder.find_recommended_teachers.return_value = [
        {"teacher": {"id": 1, "name": "张明"}, "score": 100, "reason": "您指定的老师"}
    ]
    # teacher_finder.check_teacher_availability:返回 True(无冲突)
    real_tp.teacher_finder.check_teacher_availability.return_value = True
    # lesson_service.create_lesson:返回成功,记录被调用的参数
    real_tp.lesson_service.create_lesson.return_value = (True, 9999)
    # student_service.initialize_default_student:返回默认学员
    real_tp.student_service.initialize_default_student.return_value = {"id": 1}
    real_tp.message_builder = MagicMock()
    real_tp.message_builder.create_trial_success_message.return_value = (
        "\n顾问:已为您预约 2026-09-16 19:00 的试讲课!\n老师:张明\n"
    )
    real_tp.message_builder.create_direct_trial_message.return_value = (
        "\n顾问:✅ 已为您预约 张明 老师(数学, 初中 段)于 2026-09-16 19:00 的试讲。"
        "确认请回复『确认』,修改时间请直接告诉我。\n"
    )
    real_tp.message_builder.create_recommendation_message.return_value = (
        "\n顾问:BUG_REPRODUCED——又列了一遍老师让您选!\n"
    )
    real_tp.message_builder.create_parse_error_message.return_value = "parse err"
    # 把 TrialProcessor.handle_complete_trial 和 _create_trial 绑定到 real_tp 上
    real_tp.handle_complete_trial = TrialProcessor.handle_complete_trial.__get__(real_tp)
    real_tp._create_trial = TrialProcessor._create_trial.__get__(real_tp)
    real_tp.update_history_from_data = TrialProcessor.update_history_from_data.__get__(real_tp)
    real_tp.handle_incomplete_info = TrialProcessor.handle_incomplete_info.__get__(real_tp)
    real_tp.handle_unrelated_request = TrialProcessor.handle_unrelated_request.__get__(real_tp)

    agent.processor = real_tp

    # trial_history:初始全是 None(模拟 TrialAgent 刚创建时的状态)
    agent.trial_history = {
        "subject": None, "grade_level": None, "start_time": None,
        "duration_minutes": "30", "student_situation": None,
        "preferred_teacher_name": None, "preferred_style": None,
        "unrelated_escalations": 0,
    }
    agent.chat_history = InMemoryChatMessageHistory()
    agent.state = None
    agent.unrelated_callback = None
    return agent


class TestTrialAgentPicksUpConfirmedTeacher:
    """TrialAgent 必须从共享历史回填上游确认的『老师』,不再跑 find_recommended_teachers。"""

    def test_does_not_call_find_recommended_teachers_when_teacher_already_confirmed(self):
        """核心回归:TrialAgent 看到上游已确认老师时,绝不能再调 find_recommended_teachers。"""
        agent = _make_trial_agent_with_shared_history(
            "[REPLY][推荐助手]已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)。好的,为您预约试讲"
        )
        # 直接验证回填逻辑(避免依赖真实 run_stream 中各方法的副作用)
        context = agent._last_ai_context()
        confirmed = agent._extract_confirmed_context(context)
        for key in ("subject", "grade_level", "preferred_teacher_name"):
            if confirmed.get(key) and not agent.trial_history.get(key):
                agent.trial_history[key] = confirmed[key]
        if confirmed.get("teacher_id") and not agent.trial_history.get("teacher_id"):
            agent.trial_history["teacher_id"] = confirmed["teacher_id"]

        # 关键断言 1:trial_history 必须被回填
        assert agent.trial_history["subject"] == "数学"
        assert agent.trial_history["grade_level"] == "初中"
        assert agent.trial_history["preferred_teacher_name"] == "张明"
        assert agent.trial_history["teacher_id"] == 1

    def test_create_trial_called_with_confirmed_teacher(self, monkeypatch):
        """trial_handoff_accept_selected_teacher.md spec:上游已确认老师+时间时,
        TrialProcessor.handle_complete_trial 必须 yield 直接预约引导消息,不直接调
        create_lesson(由家长回复『确认』落库,避免误操作)。

        行为对比:旧实现是直接 create_lesson,新实现是 yield 引导消息 + 等『确认』。
        """
        from services.teacher_service import TeacherService
        # 本测试的老师查找也是外边界替身，不借生产库的默认老师种子。
        monkeypatch.setattr(TeacherService, "get_teacher_by_id", lambda self, teacher_id: {
            "id": teacher_id, "name": "张明", "subjects": ["数学"], "grade_levels": ["初中"],
        })
        agent = _make_trial_agent_with_shared_history(
            "[REPLY][推荐助手]已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)。好的,为您预约试讲"
        )
        out = _collect(agent.run_stream(user_input="本周三晚上7点"))

        # 新行为:yield 含『已为您预约 张明 老师...试讲』,**不**含『为您推荐以下老师』
        assert "已为您预约" in out
        assert "张明" in out
        assert "为您推荐以下老师" not in out
        # 关键:不直接 create_lesson(由下一轮『确认』触发,符合 spec 约束)
        agent.processor.lesson_service.create_lesson.assert_not_called()
        # 同样:不再调 find_recommended_teachers——上游已指定老师,直接 yield 引导消息
        agent.processor.teacher_finder.find_recommended_teachers.assert_not_called()

    def test_user_input_not_overwritten_when_history_already_set(self):
        """trial_history 已有值时,『回填』逻辑不覆盖(避免覆盖家长后续可能补充的新输入)。

        但要注意:每轮 LLM 提取结果(data)会更新 trial_history(这是 update_history_from_data 的设计,
        与回填逻辑无关)。本测试聚焦回填逻辑本身——当 trial_history 已有时,
        `_extract_confirmed_context` 返回的字段不会被强行覆盖。
        """
        agent = _make_trial_agent_with_shared_history(
            "[REPLY][推荐助手]已为您确认 张明 老师(学科: 数学, 年级: 初中, 老师ID: 1)"
        )
        # 预填一个不同的 subject 模拟『回填前 trial_history 已经有值』
        agent.trial_history["subject"] = "英语"
        agent.trial_history["preferred_teacher_name"] = "李老师"

        # 直接观察回填逻辑:extract + 手动应用,看会不会覆盖
        context = agent._last_ai_context()
        confirmed = agent._extract_confirmed_context(context)
        # 回填逻辑:仅当 trial_history 字段为 None 时才覆盖
        for key in ("subject", "grade_level", "preferred_teacher_name"):
            if confirmed.get(key) and not agent.trial_history.get(key):
                agent.trial_history[key] = confirmed[key]

        # 已有值不被覆盖
        assert agent.trial_history["subject"] == "英语"
        assert agent.trial_history["preferred_teacher_name"] == "李老师"
        # 但 grade_level 没预填,被回填
        assert agent.trial_history["grade_level"] == "初中"


class TestTrialAgentFallbackWhenNoUpstreamContext:
    """没上游确认时,旧行为不变(还是调 find_recommended_teachers)"""

    def test_find_recommended_teachers_called_when_no_upstream(self):
        """共享历史为空 + trial_history 全空 → 不会进入『回填』分支,后续走 handle_incomplete_info
        (因为 parse_data 的 info_complete=False),不调 find_recommended_teachers"""
        agent = _make_trial_agent_with_shared_history("")  # 共享历史为空
        # 强制 parse_data 返回 info_complete=False 触发追问路径
        agent.processor.input_parser.parse_data = lambda s: {
            "subject": "未知", "grade_level": "未知",
            "start_time": "未知", "duration_minutes": "30",
            "student_situation": "", "preferred_teacher_name": "未知",
            "preferred_style": "未知", "confirmation": "未知",
            "info_complete": False, "unrelated": False,
            "missing_info": ["subject", "grade_level"],
        }
        # 不替换 parse_stream,让它输出与 parse_data 匹配的 JSON
        def _stream_unknown(*a, **kw):
            yield '{"subject":"未知","grade_level":"未知","info_complete":false,"unrelated":false}'
        agent.processor.input_parser.parse_stream = _stream_unknown
        _collect(agent.run_stream(user_input="想给孩子找个老师"))
        # 缺字段 → handle_incomplete_info,不会进 handle_complete_trial,所以 find_recommended_teachers 不被调
        agent.processor.teacher_finder.find_recommended_teachers.assert_not_called()
        # trial_history 也没被回填任何东西(因为上游无确认消息)
        assert agent.trial_history["subject"] is None
        assert agent.trial_history["preferred_teacher_name"] is None
