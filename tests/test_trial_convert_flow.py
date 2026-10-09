"""
TrialAgent 对话链路『试讲 → 转正式课』测试(trial-convert-followup spec)

背景:家长预约试讲成功后(系统回『确认请回复确认』),输入『试讲觉得不错,就要他俩』
时旧代码走 TrialInputParser → LLM 把这种话抽成 unrelated → 报『抱歉没理解您的话』。
根因:8 个意图里没有『试讲反馈/转正式课』,TrialAgent 也没有处理『满意/继续上』的分支。

修复(trial-convert-followup spec):
- P0: trial_processor 加模块级 _is_convert_intent(纯关键字,无 LLM);
      TrialAgent.pending_trial 分支在 _is_simple_confirm 未命中后、走 normal 重抽前,
      命中 convert 意图 → _handle_convert_intent(建试讲 Lesson + 引导转化 + 进 awaiting_convert)。
- P1: FlowState 加 TRIAL_CONVERT + transition_to_trial_convert; route_to_trial_convert
      (不带首轮分流文案); ClassificationProcessor TRIAL_CONVERT 态跳过分类器直转 TrialAgent;
      TrialAgent 入口加 awaiting_convert 分支:继续/满意→推荐套餐,选套餐→转正式课,流失→兜底。

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
import time as _time
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

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
    _is_convert_intent,
    _is_positive_continue,
    _is_declined_convert_intent,
    _chinese_digit_to_int,
    _is_simple_confirm,
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
# A. _is_convert_intent 单元测试(纯关键字,无 LLM)
# ===========================================================================

class TestIsConvertIntent:
    """_is_convert_intent 判定『试讲满意/想转正式课』的正向转化意图"""

    # 正向
    def test_positive(self):
        for s in (
            "试讲不错", "试讲觉得不错", "试讲满意", "试讲挺好", "讲得不错",
            "很满意", "就要他", "就要她", "就他了", "要这位老师", "喜欢这位老师",
            "转正式课", "转成正式", "报正式课", "长期学", "继续上", "想继续",
            "跟他上", "跟她学",
        ):
            assert _is_convert_intent(s), f"应判 True: {s!r}"

    # 含"不/没"的正项短语也是正项(先判正向再判否定,顺序不能反)
    def test_positive_contains_neg_char(self):
        assert _is_convert_intent("试讲不错")
        assert _is_convert_intent("讲得不错")
        assert _is_convert_intent("试讲觉得不错")

    # 否定
    def test_negative(self):
        for s in ("试讲一般", "不满意", "不要", "不太满意", "先不上了", "再想想"):
            assert not _is_convert_intent(s), f"应判 False: {s!r}"

    # 无关
    def test_unrelated(self):
        for s in ("今天天气怎么样", "确认", "改成周三", "你好"):
            assert not _is_convert_intent(s), f"应判 False: {s!r}"

    def test_empty(self):
        assert not _is_convert_intent("")
        assert not _is_convert_intent(None)
        assert not _is_convert_intent("   ")


class TestConvertHelpers:
    """转化续轮辅助判定 + 中文数字解析"""

    def test_is_positive_continue(self):
        for s in ("想继续上", "愿意", "可以", "不错", "满意"):
            assert _is_positive_continue(s), f"应判 True: {s!r}"
        assert not _is_positive_continue("先不上了")
        assert not _is_positive_continue("")

    def test_is_declined_convert_intent(self):
        for s in ("先不上了", "不要", "算了", "再考虑", "先看看", "不用了"):
            assert _is_declined_convert_intent(s), f"应判 True: {s!r}"
        for s in ("试讲不错，想继续", "1", "套餐1"):
            assert not _is_declined_convert_intent(s), f"应判 False: {s!r}"

    def test_chinese_digit_to_int(self):
        assert _chinese_digit_to_int("1") == 1
        assert _chinese_digit_to_int("一") == 1
        assert _chinese_digit_to_int("三") == 3
        assert _chinese_digit_to_int("十") == 10
        assert _chinese_digit_to_int("十二") == 12
        assert _chinese_digit_to_int("二十三") == 23
        assert _chinese_digit_to_int("") == 0


# ===========================================================================
# B. 预约成功 + 输入『试讲觉得不错，就要他俩』→ 建试讲 + 引导转化,不报『没理解』
# ===========================================================================

class TestConvertIntentCreatesTrialAndGuides:
    """pending_trial 存在 + 输入转化意图 → _handle_convert_intent:
    建试讲 Lesson + yield 成功消息 + 引导转化 + awaiting_convert=True。"""

    def test_convert_intent_not_simple_confirm(self):
        """『试讲觉得不错,就要他俩』不应被当简单确认(也不该走 confirm 落库的旧消息)。"""
        assert not _is_simple_confirm("试讲觉得不错,就要他俩")
        assert _is_convert_intent("试讲觉得不错,就要他俩")

    def test_b_creates_trial_and_guides(self):
        pending = _make_pending_trial()
        agent = _make_agent(pending=pending)

        out = _collect(agent.run_stream(user_input="试讲觉得不错，就要他俩"))

        # ❌ 不再『没理解』
        assert "没理解" not in out
        # ✅ 建试讲成功消息 + 转化引导
        assert "试讲已成功预约" in out
        assert "试讲结束后告诉我感受" in out
        assert "转正式课" in out
        # ✅ 真正落了试讲 Lesson(create_lesson 被调,lesson_type=trial)
        agent.processor.lesson_service.create_lesson.assert_called_once()
        call_kwargs = agent.processor.lesson_service.create_lesson.call_args.kwargs
        assert call_kwargs.get("lesson_type") == "trial"
        assert call_kwargs.get("status") == "confirmed"
        assert call_kwargs.get("price") == 0.0
        # ✅ pending_trial 已清 + awaiting_convert 置位
        assert "pending_trial" not in agent.trial_history
        assert agent.trial_history.get("awaiting_convert") is True
        assert agent.trial_history.get("convert_teacher_id") == pending["teacher_id"]
        assert agent.trial_history.get("convert_subject") == pending["subject"]
        assert agent.trial_history.get("convert_lesson_id") == 9999
        # ✅ state 切到 TRIAL_CONVERT
        assert agent.state.value == FlowState.TRIAL_CONVERT


# ===========================================================================
# C. awaiting_convert + 输入『试讲不错,想继续』→ 推荐套餐列表
# ===========================================================================

FAKE_PACKAGES = [
    {"id": 1, "name": "初中数学 12 周套餐", "subject": "数学", "grade_level": "初中",
     "total_lessons": 12, "duration_minutes": 60, "price_total": 7200.0,
     "price_per_lesson": 600.0, "teacher_id": 1,
     "weekly_schedule": [{"day_of_week": 3, "time": "19:00"}],
     "match_score": 90.0, "match_reason": "针对薄弱学科数学"},
]


class TestRecommendPackages:
    """家长说『试讲不错,想继续』→ 推荐正式课套餐(Top-3,含课时/价格)。"""

    def test_c_recommends_packages(self):
        agent = _make_agent(awaiting_convert=True)

        with patch("agents.trial.conversion_logic.ConversionLogic") as CL:
            CL.return_value.recommend_packages.return_value = FAKE_PACKAGES
            out = _collect(agent.run_stream(user_input="试讲不错，想继续"))

        # ✅ 输出推荐列表:含套餐名 + 课时 + 价格
        assert "初中数学 12 周套餐" in out
        assert "课时" in out
        assert "¥" in out or "价格" in out or "7200" in out
        assert "推荐" in out
        # ✅ 套餐列表缓存供下轮选择
        assert agent.trial_history.get("convert_packages") == FAKE_PACKAGES
        # ✅ unaffected awaiting_convert 仍 True
        assert agent.trial_history.get("awaiting_convert") is True


# ===========================================================================
# D. 选套餐『1』→ 创建正式系列课(schedule + series + lessons 落库)
# ===========================================================================

@pytest.fixture
def shared_db(monkeypatch, tmp_path):
    """把本测试内所有 DatabaseRouter 指向同一个临时 sqlite 文件(共享数据)。

    sqlite:///:memory: 每次连接都是独立空库,无法跨 DatabaseRouter 实例共享,
    故用单个临时文件让 seed 与后续查询(FK、teacher availability)共享同一份数据。
    """
    db_url = f"sqlite:///{tmp_path.as_posix()}/convert_shared.db"
    import db.db_router as drmod
    orig_init = drmod.DatabaseRouter.__init__

    def _patched(self, db_path="__SHARED__"):
        if db_path == "__SHARED__":
            db_path = db_url
        orig_init(self, db_path=db_path)

    monkeypatch.setattr(drmod.DatabaseRouter, "__init__", _patched)

    router = drmod.DatabaseRouter()
    teacher_id = router.teacher_repo.add_teacher(
        name="周敏", gender="女", subjects=["数学"], grade_levels=["初中"],
        teaching_style="温和", years_experience=5, hourly_rate=250.0,
        rating=4.9, is_active=True, trial_available=True,
        highlights=["耐心"],
        availability=[
            {"day": "monday", "slots": ["19:00-21:00"]},
            {"day": "tuesday", "slots": ["19:00-21:00"]},
            {"day": "wednesday", "slots": ["19:00-21:00"]},
            {"day": "thursday", "slots": ["19:00-21:00"]},
            {"day": "friday", "slots": ["19:00-21:00"]},
            {"day": "saturday", "slots": ["09:00-21:00"]},
            {"day": "sunday", "slots": ["09:00-21:00"]},
        ],
    )
    package_id = router.package_repo.add_package(
        name="初中数学 12 周套餐", teacher_id=teacher_id, subject="数学",
        grade_level="初中", total_lessons=12, duration_minutes=60,
        price_total=7200.0, price_per_lesson=600.0,
        weekly_schedule=[{"day_of_week": 3, "time": "19:00"}],
        validity_days=120, is_active=True,
    )
    parent_id = router.student_repo.get_or_create_default_parent()["id"]
    student_id = router.student_repo.add_student(
        parent_id=parent_id, name="小明", grade_level="初中",
        weak_subjects=["数学"], learning_goals=["数学进步"],
        preferred_style="温和", school="朝阳中学",
    )
    return {"teacher_id": teacher_id, "package_id": package_id,
            "student_id": student_id}


class TestConvertToSeries:
    """家长输入套餐序号『1』→ 真正创建正式系列课,DB 侧生成 series + lessons。"""

    def test_d_creates_series_and_lessons(self, shared_db):
        from services.lesson_service import LessonService
        from db.db_router import DatabaseRouter

        student_id = shared_db["student_id"]
        # 用真实套餐 dict 作为已推荐套餐(与 DB 一致,含 teacher_id/id/weekly 等)
        real_pkg = DatabaseRouter().package_repo.get_package_by_id(
            shared_db["package_id"]
        )
        agent = _make_agent(
            awaiting_convert=True, convert_packages=[real_pkg],
            convert_lesson_id=9999, convert_teacher_id=shared_db["teacher_id"],
            student_id=student_id,
        )

        out = _collect(agent.run_stream(user_input="1"))

        # ✅ 输出成功消息(含正式课/系列/课时字样)
        assert "正式课" in out
        assert "系列" in out
        assert "课时" in out
        assert "套餐: 初中数学" in out or "初中数学" in out
        # ✅ 创建完成即清转化态
        assert agent.trial_history.get("awaiting_convert") is None
        # ✅ DB 侧生成 series + lessons(可断言 LessonService 查询)
        svc = LessonService()
        series_lessons = svc.get_lessons_by_student(
            student_id, lesson_type="series"
        )
        assert len(series_lessons) >= 1, "应至少创建 1 节正式系列课"
        for l in series_lessons:
            assert l.get("lesson_type") == "series"
            assert l.get("status") == "confirmed"
            assert l.get("package_id") == shared_db["package_id"]
        series_id = series_lessons[0]["series_id"]
        assert series_id is not None
        assert svc.get_series(series_id) is not None
        # 套餐推荐后选套餐,convert_packages 应在转化后清空
        assert agent.trial_history.get("convert_packages") is None

    def test_select_by_package_name(self, shared_db):
        """支持按套餐名选择(而不是序号)。"""
        from db.db_router import DatabaseRouter
        from services.lesson_service import LessonService

        student_id = shared_db["student_id"]
        real_pkg = DatabaseRouter().package_repo.get_package_by_id(
            shared_db["package_id"]
        )
        agent = _make_agent(
            awaiting_convert=True, convert_packages=[real_pkg],
            convert_lesson_id=9999, convert_teacher_id=shared_db["teacher_id"],
            student_id=student_id,
        )

        out = _collect(agent.run_stream(user_input="就选初中数学 12 周套餐"))

        assert "正式课" in out
        assert len(LessonService().get_lessons_by_student(
            student_id, lesson_type="series"
        )) >= 1


# ===========================================================================
# E. awaiting_convert + 输入『先不上了』→ 流失兜底,清 awaiting_convert,不崩
# ===========================================================================

class TestDeclineClearsConvert:
    def test_e_churn_fallback(self):
        agent = _make_agent(awaiting_convert=True)

        out = _collect(agent.run_stream(user_input="先不上了"))

        # ✅ 流失兜底消息 + 清转化态
        assert "有需要随时找我" in out
        assert agent.trial_history.get("awaiting_convert") is None
        assert agent.trial_history.get("convert_teacher_id") is None
        # ❌ 不报错、不推荐套餐
        assert "套餐" not in out


# ===========================================================================
# 主路由:TRIAL_CONVERT 态跳过分类器,直接转 TrialAgent
# ===========================================================================

class TestStateManagerTrialConvert:
    def test_transition_and_detection(self):
        sm = StateManager()
        assert sm.is_in_trial_convert_flow() is False
        sm.transition_to_trial_convert()
        assert sm.get_current_state() == FlowState.TRIAL_CONVERT
        assert sm.is_in_trial_convert_flow() is True
        assert sm.should_classify() is False
        assert sm.get_state_description() == "试讲转化态"


class TestRouteToTrialConvert:
    def test_route_skips_classification_thought(self):
        agent = _FakeTrialAgent()
        router = AgentRouter(trial_agent=agent, state_manager=MagicMock())
        out = _run(_drain(router.route_to_trial_convert("试讲不错，想继续")))
        # 不带首轮分流文案(与 route_to_trial 对比)
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" not in out
        assert agent.calls == ["试讲不错，想继续"]

    def test_route_without_agent_yields_error(self):
        router = AgentRouter(trial_agent=None, state_manager=MagicMock())
        out = _run(_drain(router.route_to_trial_convert("任意")))
        assert "[ERROR]" in out

    def test_route_differs_from_route_to_trial(self):
        agent = _FakeTrialAgent()
        router = AgentRouter(trial_agent=agent, state_manager=MagicMock())
        out1 = _run(_drain(router.route_to_trial("任意")))
        out2 = _run(_drain(router.route_to_trial_convert("任意")))
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" in out1
        assert "[THOUGHT][归类机器人]" not in out2


class TestClassificationProcessorTrialConvertBypass:
    def test_stream_skips_classifier_in_trial_convert_state(self):
        sm = StateManager()
        sm.transition_to_trial_convert()
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
        prior = classifier.call_count
        out = _collect(processor.process_task_stream("试讲不错，想继续"))
        assert classifier.call_count == prior
        assert "[THOUGHT][归类机器人] 这是一个试讲预约请求" not in out

    def test_sync_skips_classifier_in_trial_convert_state(self):
        sm = StateManager()
        sm.transition_to_trial_convert()
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
        prior = classifier.call_count
        result = _run(processor.process_task_sync("1"))
        assert classifier.call_count == prior
        assert "[THOUGHT][归类机器人]" not in result


# ===========================================================================
# 辅助工具
# ===========================================================================

class _FakeTrialAgent:
    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None, **kw):
        self.calls.append(user_input)
        yield "[REPLY][试讲顾问]试讲转化续轮结果"


class _CounterClassifier:
    def __init__(self, return_category="unrelated"):
        self.call_count = 0
        self.return_category = return_category

    async def classify_task(self, task):
        self.call_count += 1
        return self.return_category


class _FakeParser:
    def parse_stream(self, user_input, chat_history, context=""):
        yield '{"subject":"数学","grade_level":"初中","start_time":"2026-09-16 19:00","info_complete":true}'

    def parse_data(self, _s):
        return {
            "subject": "数学", "grade_level": "初中",
            "start_time": "2026-09-16 19:00", "duration_minutes": "30",
            "student_situation": "", "preferred_teacher_name": "周敏",
            "preferred_style": "未知", "confirmation": "未知",
            "info_complete": True, "unrelated": False, "missing_info": [],
        }


def _make_pending_trial() -> dict:
    return {
        "teacher_id": 1,
        "teacher_name": "周敏",
        "subject": "数学",
        "grade_level": "初中",
        "time_str": "2026-09-16 19:00",
        "time_iso": "2026-09-16T19:00:00",
        "duration_minutes": 30,
        "created_at": _time.time(),
    }


def _make_agent(pending=None, awaiting_convert=False, convert_packages=None,
                convert_lesson_id=9999, convert_teacher_id=1, student_id=1) -> TrialAgent:
    """绕过 __init__(避免真实 LLM),构造 TrialAgent-like 实例。

    主要 mock:
    - input_parser.parse_stream / parse_data:固定 JSON(防御正常解析路径)
    - processor.lesson_service.create_lesson:返回成功 + lesson_id
    - processor.student_service.initialize_default_student:返回默认学员
    - state:Mock(允许 .value 赋值)
    """
    agent = object.__new__(TrialAgent)
    agent.session_id = "test-convert-session"

    class _StubProcessor:
        pass

    real_tp = _StubProcessor()
    real_tp.input_parser = _FakeParser()
    real_tp.teacher_finder = MagicMock()
    real_tp.teacher_finder.find_recommended_teachers.return_value = []
    real_tp.teacher_finder.check_teacher_availability.return_value = True
    real_tp.lesson_service = MagicMock()
    real_tp.lesson_service.create_lesson.return_value = (True, convert_lesson_id)
    real_tp.student_service = MagicMock()
    real_tp.student_service.initialize_default_student.return_value = {"id": student_id}
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
        "start_time": "2026-09-16 19:00", "duration_minutes": "30",
        "student_situation": None,
        "preferred_teacher_name": "周敏", "preferred_style": None,
        "teacher_id": 1,
        "unrelated_escalations": 0,
    }
    if pending is not None:
        agent.trial_history["pending_trial"] = pending
    if awaiting_convert:
        agent.trial_history["awaiting_convert"] = True
        agent.trial_history["convert_teacher_id"] = convert_teacher_id
        agent.trial_history["convert_subject"] = "数学"
        agent.trial_history["convert_lesson_id"] = convert_lesson_id
    if convert_packages is not None:
        agent.trial_history["convert_packages"] = convert_packages

    agent.chat_history = MagicMock()
    agent.chat_history.messages = []
    agent.chat_history.add_message = lambda m: None
    agent.chat_history.clear = lambda: None

    agent.state = MagicMock()
    agent.state.value = "classify"

    agent.unrelated_callback = None
    agent.context_manager = None
    agent.session_chat_history = None
    agent.finished = False
    agent._record_shared_ai = lambda c: None
    return agent