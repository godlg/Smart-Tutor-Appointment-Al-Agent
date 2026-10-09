"""
试讲链路三修复测试
  A. 试讲确认前冲突预检(TrialProcessor.handle_complete_trial direct 分支)
  B. 相对时间解析(config/time_config.py parse_datetime 兜底增强)
  C. 试讲流程内档期查询(TrialProcessor._handle_availability_query)

约束(与 tests/test_reschedule_routing.py 一致):
  - 无 pytest-asyncio,异步用例统一用 _run(_collect) 同步包装
  - 构造测试替身用 object.__new__ 跳过 __init__,避免连真实 LLM/DB
  - mock TeacherService / lesson_repo / teacher_finder,不触碰生产数据库
  - B 用可注入基准日期(base_date=_parse_relative_datetime 的入参),不依赖真实当前日期
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trial.trial_processor import (  # noqa: E402
    TrialProcessor,
    TrialMessageBuilder,
    _is_availability_query,
)
from config.time_config import TimeConfig  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


def _build_processor(teacher_finder=None):
    """构造一个不碰 LLM、不碰 DB 的 TrialProcessor 替身(绑真实 handler 方法)。

    跳过 __init__,手挂依赖:teacher_finder(可 mock)/lesson_service(挂 lesson_repo
    用于 C)/message_builder(真实,便于断言文案)。
    """
    tp = object.__new__(TrialProcessor)
    tp.input_parser = None
    tp.teacher_finder = teacher_finder if teacher_finder is not None else MagicMock()
    tp.message_builder = TrialMessageBuilder()
    tp.lesson_service = MagicMock()
    tp.student_service = MagicMock()
    tp.handle_complete_trial = TrialProcessor.handle_complete_trial.__get__(tp)
    tp._handle_availability_query = TrialProcessor._handle_availability_query.__get__(tp)
    return tp


def _finder(available: bool):
    ff = MagicMock()
    ff.check_teacher_availability.return_value = available
    return ff


ZHANG = {
    "id": 1,
    "name": "张明",
    "subjects": ["数学"],
    "grade_levels": ["初中"],
}


def _patch_teacher_service(monkeypatch, teacher=ZHANG):
    """把 services.teacher_service.TeacherService 换成可按 id/name 返回固定老师的类。

    生产代码调用 TeacherService()(实例化),故此处必须是【可调用的类】。
    """
    fake_cls = type("_TS", (), {
        "get_teacher_by_id": staticmethod(lambda tid: teacher),
        "get_teacher_by_name": staticmethod(lambda name: teacher),
    })
    monkeypatch.setattr(
        "services.teacher_service.TeacherService",
        fake_cls,
    )


def _trial_history(**overrides):
    th = {
        "subject": "数学",
        "grade_level": "初中",
        "start_time": "2026-09-18 19:00",
        "duration_minutes": "30",
        "student_situation": None,
        "preferred_teacher_name": None,
        "preferred_style": None,
        "teacher_id": 1,
        "unrelated_escalations": 0,
    }
    th.update(overrides)
    return th


# ===========================================================================
# A. 试讲确认前冲突预检
# ===========================================================================

class TestConflictPrecheck:
    """direct trial 分支发『已为您预约…确认』之前,先做老师档期预检。

    冲突 → 当场回复换时间:不承诺成功、不写 pending_trial/confirmed_*;
    可用 → 原行为不变:发引导消息 + 写 pending_trial + 存 confirmed_*。
    """

    def test_conflict_does_not_promise_success(self, monkeypatch):
        # Arrange:teacher_finder 判定冲突
        tp = _build_processor(teacher_finder=_finder(available=False))
        _patch_teacher_service(monkeypatch)
        th = _trial_history()

        # Act
        out = _collect(tp.handle_complete_trial(th, "s1"))

        # Assert:当场换时间,不承诺成功
        assert "张明老师在该时段已有课程" in out
        assert "已为您预约" not in out
        assert "确认请回复" not in out
        # 不写 pending_trial / confirmed_* / start_time_parsed / duration_minutes_int
        assert "pending_trial" not in th
        assert "confirmed_teacher_id" not in th
        assert "confirmed_teacher" not in th
        assert "start_time_parsed" not in th
        assert "duration_minutes_int" not in th

    def test_conflict_message_exact_format(self, monkeypatch):
        tp = _build_processor(teacher_finder=_finder(available=False))
        _patch_teacher_service(monkeypatch)
        th = _trial_history()

        out = _collect(tp.handle_complete_trial(th, "s1"))

        assert out.startswith("[REPLY][试讲顾问]")
        assert "已有课程" in out
        assert "请重新选择时间" in out

    def test_available_preserves_original_behavior(self, monkeypatch):
        # Arrange:teacher_finder 判定可用
        tp = _build_processor(teacher_finder=_finder(available=True))
        _patch_teacher_service(monkeypatch)
        th = _trial_history()

        # Act
        out = _collect(tp.handle_complete_trial(th, "s1"))

        # Assert:原行为不变——发引导消息 + 写 pending_trial + 存 confirmed_*
        assert "确认请回复" in out
        assert "pending_trial" in th
        assert th["pending_trial"]["teacher_id"] == 1
        assert th["pending_trial"]["teacher_name"] == "张明"
        assert th["confirmed_teacher_id"] == 1
        assert th["start_time_parsed"] is not None
        assert th["duration_minutes_int"] == 30

    def test_precheck_receives_naive_datetime(self, monkeypatch):
        """tz 坑:start_time 是 tz-aware(Beijing),传给 check_teacher_availability 前
        必须 strip tzinfo,否则内部与 naive 比较会抛异常。"""
        ff = _finder(available=True)
        tp = _build_processor(teacher_finder=ff)
        _patch_teacher_service(monkeypatch)
        th = _trial_history()

        _collect(tp.handle_complete_trial(th, "s1"))

        assert ff.check_teacher_availability.called
        args = ff.check_teacher_availability.call_args.args
        assert args[0] == 1
        assert args[1].hour == 19
        assert args[1].tzinfo is None
        assert args[2] == 30


# ===========================================================================
# B. 相对时间解析
# ===========================================================================

class TestRelativeTimeParsing:
    """用可注入基准日期(2026-09-16 周三,与 spec 一致),不依赖真实当前日期。

    基准:2026-09-16 是周三 → 本周五=09-18、本周六=09-19。
    """

    BASE = datetime(2026, 9, 16)

    @staticmethod
    def _strip(dt):
        return dt.replace(tzinfo=None)

    def test_this_week_friday_evening(self):
        dt = TimeConfig._parse_relative_datetime("这个星期周五晚上", self.BASE)
        assert dt is not None
        assert self._strip(dt) == datetime(2026, 9, 18, 19, 0)

    def test_bare_friday_evening(self):
        dt = TimeConfig._parse_relative_datetime("周五晚上", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 18, 19, 0)

    def test_this_saturday_morning(self):
        dt = TimeConfig._parse_relative_datetime("这周六上午", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 19, 9, 0)

    def test_tomorrow_evening_7(self):
        dt = TimeConfig._parse_relative_datetime("明天晚上7点", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 17, 19, 0)

    def test_afternoon_explicit_hour(self):
        dt = TimeConfig._parse_relative_datetime("下周一下午3点", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 21, 15, 0)

    def test_past_weekday_rolls_to_next_week(self):
        """周三说『周一』——本周一已过,顺延下周(09-21)。"""
        dt = TimeConfig._parse_relative_datetime("周一上午", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 21, 9, 0)

    def test_default_period_hour(self):
        dt = TimeConfig._parse_relative_datetime("今天下午", self.BASE)
        assert self._strip(dt) == datetime(2026, 9, 16, 14, 0)

    def test_unparseable_returns_none(self):
        assert TimeConfig._parse_relative_datetime("太远了不确定", self.BASE) is None
        assert TimeConfig._parse_relative_datetime("", self.BASE) is None
        assert TimeConfig._parse_relative_datetime(None, self.BASE) is None

    def test_parse_datetime_strict_still_tz_aware(self):
        """parse_datetime 签名/返回不变:绝对时间仍返回带 Beijing tz 的 datetime。"""
        dt = TimeConfig.parse_datetime("2026-09-18 19:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.replace(tzinfo=None) == datetime(2026, 9, 18, 19, 0)


class TestStartTimeFallback:
    """B 兜底:LLM 字段提取失败(返回 start_time=未知/空)时,
    _fallback_start_time_from_input 直接用 parse_datetime 解析家长原始输入。"""

    def _this_week_friday(self):
        """动态计算本周五 19:00（fallback 以真实当前日期为基准）"""
        today = TimeConfig.today()
        friday = today + timedelta(days=(4 - today.weekday()) % 7)
        return friday.replace(hour=19, minute=0).strftime("%Y-%m-%d %H:%M")

    def test_fills_when_llm_missing(self):
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "未知", "subject": "数学", "grade_level": "初中"}
        ok = _fallback_start_time_from_input(data, "这个星期周五晚上")
        assert ok
        assert data["start_time"] == self._this_week_friday()

    def test_fills_when_llm_partial_garbage(self):
        """残缺值"2026-09-18 未知"(日期对、时间"未知")不该阻止原文兜底。"""
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "2026-09-18 未知", "subject": "数学", "grade_level": "初中"}
        ok = _fallback_start_time_from_input(data, "这个星期周五晚上")
        assert ok
        assert data["start_time"] == self._this_week_friday()

    def test_noop_when_normal_absolute_time(self):
        """raw 是完整可解析的绝对时间 → 不兜底,原样保留。"""
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "2026-09-18 19:00"}
        ok = _fallback_start_time_from_input(data, "这个星期周五晚上")
        assert not ok
        assert data["start_time"] == "2026-09-18 19:00"

    def test_noop_when_empty_user_input(self):
        """user_input 为空 → 直接返回 False,不做兜底。"""
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "2026-09-18 未知"}
        ok = _fallback_start_time_from_input(data, "")
        assert not ok
        assert data["start_time"] == "2026-09-18 未知"

    def test_noop_when_partial_garbage_and_user_unparseable(self):
        """残缺值 + 原文也 parse 不了 → 返回 False,保留下游"无法识别时间"兜底。"""
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "2026-09-18 未知"}
        ok = _fallback_start_time_from_input(data, "太远了不确定")
        assert not ok
        assert data["start_time"] == "2026-09-18 未知"

    def test_fills_bare_relative(self):
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": ""}
        ok = _fallback_start_time_from_input(data, "周五晚上")
        assert ok
        assert data["start_time"] == self._this_week_friday()

    def test_noop_when_already_present(self):
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "2026-09-20 10:00"}
        ok = _fallback_start_time_from_input(data, "这个星期周五晚上")
        assert not ok
        assert data["start_time"] == "2026-09-20 10:00"

    def test_noop_when_unparseable(self):
        from agents.trial.trial_processor import _fallback_start_time_from_input
        data = {"start_time": "未知"}
        ok = _fallback_start_time_from_input(data, "随便聊聊孩子情况")
        assert not ok
        assert data["start_time"] == "未知"


# ===========================================================================
# C. 试讲流程内档期查询
# ===========================================================================

class TestAvailabilityQuery:

    def test_unit_matcher(self):
        for s in ("张明老师什么时候有空", "这老师几点没空", "老师空闲吗", "什么时间方便", "查一下档期"):
            assert _is_availability_query(s), f"应命中: {s!r}"
        for s in ("确认", "改成周五晚上", "你好", ""):
            assert not _is_availability_query(s), f"不应命中: {s!r}"

    def test_uses_confirmed_teacher(self, monkeypatch):
        tp = _build_processor()
        lesson_repo = MagicMock()
        lesson_repo.check_teacher_conflict.return_value = False
        tp.lesson_service.lesson_repo = lesson_repo
        _patch_teacher_service(monkeypatch)
        th = {"teacher_id": 1}

        out = _collect(tp._handle_availability_query("什么时候有空", th))

        assert "张明老师最近可预约" in out
        assert "可预约" in out
        assert ":00" in out  # 时间格式整点
        lesson_repo.check_teacher_conflict.assert_called()

    def test_parses_teacher_name_from_input(self, monkeypatch):
        tp = _build_processor()
        lesson_repo = MagicMock()
        lesson_repo.check_teacher_conflict.return_value = False
        tp.lesson_service.lesson_repo = lesson_repo
        _patch_teacher_service(monkeypatch)
        th = {}  # 无已确认/已选老师

        out = _collect(tp._handle_availability_query("张明老师什么时候有空", th))

        assert "张明老师最近可预约" in out

    def test_all_slots_conflict_reports_no_availability(self, monkeypatch):
        tp = _build_processor()
        lesson_repo = MagicMock()
        # 全部时段都冲突
        def _conflict(tid, start, end):
            return True
        lesson_repo.check_teacher_conflict.side_effect = _conflict
        tp.lesson_service.lesson_repo = lesson_repo
        _patch_teacher_service(monkeypatch)
        th = {"teacher_id": 1}

        out = _collect(tp._handle_availability_query("张明老师什么时候有空", th))

        assert "暂时没有空闲时段" in out
        assert "可预约:" not in out

    def test_no_teacher_guides_selection(self, monkeypatch):
        tp = _build_processor()
        tp.lesson_service.lesson_repo = MagicMock()
        # teacher 解析失败 → 不返回时段,提示选老师
        _patch_teacher_service(monkeypatch, teacher=None)
        th = {}

        out = _collect(tp._handle_availability_query("什么时候有空", th))

        assert "回复老师姓名" in out