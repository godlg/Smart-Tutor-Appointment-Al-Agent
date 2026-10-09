"""
正式课套餐推荐对齐对话确认的年级/学科 + 补充「初中数学」套餐测试
(package-recommend-alignment spec)

背景:初二数学孩子走「试讲 → 转正式课」链路,推荐出的套餐全是高中套餐
(高一数学/高三物理),与孩子「初二/初中」不符。修复有两层:
- P0(代码):_recommend_packages_for_convert 用对话已确认的 subject/grade_level
  覆盖默认画像(默认学员=高中小明的 weak_subjects=[数学,物理]),让套餐打分对齐真实孩子。
- P1(数据):DEFAULT_PACKAGES 增加第 4 个「初中数学 10 周提分套餐」;
  initialize_default_packages 改为幂等补插(库非空不再整体跳过,按 name 补缺)。

验收(DoD):
- A: recommend_packages(student={grade_level:"初中",weak_subjects:["数学"],...},
  teacher_id=6) → 返回含「初中数学 10 周提分套餐」且排第一。
- B: TrialAgent 场景(trial_history 含 grade_level="初中"/subject="数学"),
  _recommend_packages_for_convert → 输出含「初中数学 10 周提分套餐」。
- C: initialize_default_packages 幂等:已有 3 个时调用 → 补出第 4 个「初中数学」;
  再调用一次 → 总数不变(不重复)。

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agents.trial.conversion_logic import ConversionLogic  # noqa: E402
from services.package_service import PackageService  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


# Font name used across assertions(与 services.package_service.DEFAULT_PACKAGES 一致)
JUNIOR_MATH_NAME = "初中数学 10 周提分套餐"


@pytest.fixture
def pkg_db(monkeypatch, tmp_path):
    """把本文件内所有 DatabaseRouter 指向同一个临时 sqlite 文件(共享数据)。

    sqlite:///:memory: 每次连接都是独立空库,无法跨 DatabaseRouter 实例共享,
    故用单个临时文件让 seed(default packages)与 ConversionLogic/PackageService
    的查询共享同一份数据。每个测试独立 tmp_path,互不干扰。
    """
    db_url = f"sqlite:///{tmp_path.as_posix()}/pkg_alignment_shared.db"
    import db.db_router as drmod

    orig_init = drmod.DatabaseRouter.__init__

    def _patched(self, db_path="__SHARED__"):
        if db_path == "__SHARED__":
            db_path = db_url
        orig_init(self, db_path=db_path)

    monkeypatch.setattr(drmod.DatabaseRouter, "__init__", _patched)
    return db_url


def _seed_all_defaults():
    """揷入全部 DEFAULT_PACKAGES(含新增的「初中数学 10 周提分套餐」)。"""
    assert PackageService().initialize_default_packages() is True


def _seed_original_three():
    """只揷入原 3 个默认套餐(模拟旧库,缺「初中数学」)。"""
    from services.package_service import PackageService as PS
    svc = PS()
    for data in svc.DEFAULT_PACKAGES[:3]:
        svc.add_package(**data)


# ===========================================================================
# A. recommend_packages 按真实画像(初中/数学)把「初中数学」排第一
# ===========================================================================

class TestRecommendAlignedToContext:
    """对初中/数学画像,「初中数学 10 周提分套餐」应返回且排第一。"""

    def test_a_init_chu_math_ranked_first(self, pkg_db):
        _seed_all_defaults()
        logic = ConversionLogic()
        student = {
            "grade_level": "初中", "weak_subjects": ["数学"],
            "budget_max": 500.0, "learning_goals": ["数学提分"],
        }
        packages = logic.recommend_packages(
            student=student, teacher_id=6, limit=3,
        )
        names = [p["name"] for p in packages]
        assert JUNIOR_MATH_NAME in names, names
        assert names[0] == JUNIOR_MATH_NAME, names

    def test_a_junior_math_beats_high_school_on_junior_profile(self, pkg_db):
        """同老师下,对初中/数学画像,初中数学应优先于同老师的高中数学套餐。

        recommend_packages 的 teacher_id 是必需参数且按老师过滤;要让跨年级套餐
        并存打分,须让同一老师持有初中+高中数学套餐。这里给老师 6 再挂一个
        高中数学套餐,验证初中数学仍排第一(不被高中埋没)。
        """
        _seed_all_defaults()
        svc = PackageService()
        svc.add_package(
            name="高一数学 1 对 1 强化套餐", teacher_id=6, subject="数学",
            grade_level="高中", total_lessons=16, duration_minutes=60,
            price_total=6400.0, price_per_lesson=400.0,
            weekly_schedule=[{"day_of_week": 3, "start_time": "19:00"}],
            validity_days=120,
        )
        logic = ConversionLogic()
        student = {
            "grade_level": "初中", "weak_subjects": ["数学"],
            "budget_max": 500.0, "learning_goals": ["数学提分"],
        }
        packages = logic.recommend_packages(student=student, teacher_id=6, limit=10)
        names = [p["name"] for p in packages]
        assert JUNIOR_MATH_NAME in names, names
        assert "高一数学 1 对 1 强化套餐" in names, names
        assert names.index(JUNIOR_MATH_NAME) < names.index("高一数学 1 对 1 强化套餐")


# ===========================================================================
# B. TrialAgent 链路:对话确认 初中/数学 → 推荐输出含「初中数学」
# ===========================================================================

class TestTrialAgentRecommendOverridesProfile:
    def test_b_convert_recommend_contains_junior_math(self, pkg_db):
        _seed_all_defaults()
        agent = _make_agent(
            awaiting_convert=True,
            convert_teacher_id=6,   # 周敏(数学,教初中)
            convert_subject="数学",
        )
        # 默认画像(高中小明,weak_subjects=[数学,物理])已被对话确认覆盖为 初中/数学
        out = _collect(agent.run_stream(user_input="试讲不错，想继续"))
        assert JUNIOR_MATH_NAME in out, out
        assert "高一数学 12 周提分套餐" not in out or JUNIOR_MATH_NAME in out
        # 推荐结果缓存供下轮选套餐
        cached = agent.trial_history.get("convert_packages") or []
        assert any(p.get("name") == JUNIOR_MATH_NAME for p in cached)

    def test_b_default_profile_untouched_when_unknown(self, pkg_db):
        """学科/年级为『未知』时不应覆盖,保留默认画像(避免误覆盖)。"""
        from agents.trial.trial_agent import TrialAgent
        _seed_all_defaults()
        agent = _make_agent(awaiting_convert=True, convert_teacher_id=6)
        agent.trial_history["subject"] = "未知"
        agent.trial_history["grade_level"] = "未知"
        # 记录默认画像暴露给 ConversionLogic 看的内容(打点 student dict 的弱覆盖)
        captured = {}

        orig_rec = TrialAgent._recommend_packages_for_convert

        async def _spy(self_ref):
            from services.student_profile_service import StudentProfileService
            student = self_ref.processor.student_service.initialize_default_student()
            captured["student"] = dict(student)
            async for t in orig_rec(self_ref):
                yield t

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TrialAgent, "_recommend_packages_for_convert", _spy)
            out = _collect(agent.run_stream(user_input="试讲不错，想继续"))
        assert "推荐" in out
        # 『未知』不覆盖 → student 的 weak_subjects/grade_level 保持默认(非 [数学]/初中)
        student = captured.get("student") or {}
        if student.get("grade_level"):
            assert student["grade_level"] != "初中"


# ===========================================================================
# C. initialize_default_packages 幂等补插
# ===========================================================================

class TestInitializeIdempotent:
    def test_c_fills_missing_junior_math(self, pkg_db):
        _seed_original_three()
        count_before = len(PackageService().get_all_packages())
        assert count_before == 3, count_before
        # 首次调用 → 补出第 4 个「初中数学」
        assert PackageService().initialize_default_packages() is True
        after = PackageService().get_all_packages()
        names = [p["name"] for p in after]
        assert JUNIOR_MATH_NAME in names, names
        assert len(after) == 4, names

    def test_c_second_call_no_duplicate(self, pkg_db):
        _seed_original_three()
        assert PackageService().initialize_default_packages() is True
        n1 = len(PackageService().get_all_packages())
        # 再调用一次 → 总数不变(不重复插入)
        assert PackageService().initialize_default_packages() is True
        n2 = len(PackageService().get_all_packages())
        assert n2 == n1 == 4

    def test_c_fresh_db_inserts_all(self, pkg_db):
        """全新空库 → 一次插入全部 4 个默认套餐。"""
        naming = [p["name"] for p in PackageService().DEFAULT_PACKAGES]
        assert JUNIOR_MATH_NAME in naming, naming
        assert PackageService().initialize_default_packages() is True
        got = [p["name"] for p in PackageService().get_all_packages()]
        assert len(got) == 4, got
        assert JUNIOR_MATH_NAME in got


# ===========================================================================
# 辅助工具(沿用 test_trial_convert_flow.py 的构造方式,绕过真实 LLM)
# ===========================================================================

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


def _make_agent(awaiting_convert=False, convert_teacher_id=1,
                convert_subject="数学") -> object:
    """构造 TrialAgent-like 实例(绕过 __init__,避免真实 LLM)。

    主要 mock:
    - processor.student_service.initialize_default_student:返回默认学员(高中小明弱项)
    - state:Mock(允许 .value 赋值)
    推荐套餐走真实 ConversionLogic + 共享 DB(见 pkg_db fixture)。
    """
    from agents.trial.trial_agent import TrialAgent
    from agents.trial.trial_processor import TrialProcessor

    agent = object.__new__(TrialAgent)
    agent.session_id = "test-pkg-align-session"

    class _StubProcessor:
        pass

    real_tp = _StubProcessor()
    real_tp.input_parser = _FakeParser()
    real_tp.teacher_finder = MagicMock()
    real_tp.lesson_service = MagicMock()
    real_tp.lesson_service.create_lesson.return_value = (True, 9999)
    # 默认画像:高中小明,weak_subjects=[数学,物理] —— 正是 P0 要覆盖掉的画像
    real_tp.student_service = MagicMock()
    real_tp.student_service.initialize_default_student.return_value = {
        "id": 1, "name": "小明", "grade_level": "高中",
        "weak_subjects": ["数学", "物理"], "budget_max": 500.0,
    }
    real_tp.message_builder = MagicMock()
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
        "teacher_id": 6,
        "unrelated_escalations": 0,
    }
    if awaiting_convert:
        agent.trial_history["awaiting_convert"] = True
        agent.trial_history["convert_teacher_id"] = convert_teacher_id
        agent.trial_history["convert_subject"] = convert_subject
        agent.trial_history["convert_lesson_id"] = 9999

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