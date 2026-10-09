"""
全局档期查询路由测试(fix-availability-global-route)

目标:新会话(CLASSIFY)直接问"张明老师什么时候有空" → 返回可用时段列表,
不再回"没有实时档期数据";老师名提取 + 未知老师引导;`trial` 态档期查询不回归。

约束(与 tests/test_trial_conflict_precheck.py 一致):
  - 无 pytest-asyncio,异步用例统一用 _run(_collect) 同步包装
  - 构造替身用 object.__new__ / 注入 fake service,不连真实 LLM/DB
  - mock TeacherService / LessonService,不触碰生产数据库
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.availability.availability_handler import (  # noqa: E402
    AvailabilityHandler,
    is_availability_query,
)
from agents.task_classification.classification_processor import (  # noqa: E402
    ClassificationProcessor,
)
from agents.task_classification.unrelated_handler import UnrelatedHandler  # noqa: E402


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
# 测试数据 / 替身
# ----------------------------------------------------------------------

# TeacherService 替身:可按名查 + 枚举全部老师
class _FakeTeacherService:
    def __init__(self, teachers):
        self._teachers = teachers
        self._by_name = {t["name"]: t for t in teachers}

    def get_teacher_by_name(self, name):
        return self._by_name.get(name)

    def get_all_teachers(self):
        return self._teachers


ZHANG = {"id": 1, "name": "张明"}
GAOYUAN = {"id": 7, "name": "高远"}
ALL_TEACHERS = [ZHANG, GAOYUAN]


class _FakeLessonService:
    """LessonService 替身:挂 lesson_repo;conflict=False 默认全部时段可用。"""

    def __init__(self, conflict=False):
        repo = MagicMock()
        repo.check_teacher_conflict.return_value = conflict
        self.lesson_repo = repo


def _processed_handler(teachers=ALL_TEACHERS, conflict=False):
    """构造一个不碰 LLM/DB 的 AvailabilityHandler(注入 fake service)。"""
    return AvailabilityHandler(
        teacher_service=_FakeTeacherService(teachers),
        lesson_service=_FakeLessonService(conflict),
    )


# ----------------------------------------------------------------------
# 1. 档期关键词规则检测
# ----------------------------------------------------------------------

class TestAvailabilityKeyword:

    def test_unit_matcher(self):
        for s in ("张明老师什么时候有空", "查一下高远老师的档期", "周敏老师什么时间段没空",
                  "什么时候方便", "时间安排如何", "什么时间有空"):
            assert is_availability_query(s), f"应命中: {s!r}"
        for s in ("确认", "改成周五晚上", "推荐个老师", "你好", ""):
            assert not is_availability_query(s), f"不应命中: {s!r}"


# ----------------------------------------------------------------------
# 2. AvailabilityHandler:老师名提取
# ----------------------------------------------------------------------

class TestExtractTeacher:

    def test_extract_by_substring_forwordy_input(self):
        """'查一下高远老师的档期' → 高远(前缀干扰不得抓成'查一下高远')"""
        h = _processed_handler()
        out = h._extract_teacher(h.teacher_service, "查一下高远老师的档期")
        assert out["name"] == "高远"

    def test_extract_by_name(self):
        h = _processed_handler()
        assert h._extract_teacher(h.teacher_service, "张明老师什么时候有空")["name"] == "张明"

    def test_extract_unknown_returns_none(self):
        h = _processed_handler()
        assert h._extract_teacher(h.teacher_service, "王五老师什么时候有空") is None


# ----------------------------------------------------------------------
# 3. AvailabilityHandler:档期查询消息
# ----------------------------------------------------------------------

class TestAvailabilitySlots:

    def test_available_slots_reply(self):
        h = _processed_handler(conflict=False)
        out = _collect(h.handle("张明老师什么时候有空"))
        assert "可预约" in out
        assert "请告诉我要哪个时间" in out
        assert ":00" in out  # 整点格式
        assert "没有实时档期" not in out

    def test_all_slots_conflict_reports_none(self):
        h = _processed_handler(conflict=True)
        out = _collect(h.handle("张明老师什么时候有空"))
        assert "暂时没有空闲时段" in out
        assert "可预约:" not in out

    def test_unknown_teacher_guides(self):
        h = _processed_handler()
        out = _collect(h.handle("王五老师什么时候有空"))
        assert "请告诉我要查询哪位老师" in out
        assert "可预约" not in out


# ----------------------------------------------------------------------
# 4. ClassificationProcessor:全局拦截(CLASSIFY 态)
# ----------------------------------------------------------------------

class _AsyncConsultationClassifier:
    """分类器固定返回 'consultation'(复现 bug:LLM 把档期问题判成咨询)。"""

    async def classify_task(self, task):
        return "consultation"


class _FakeRouter:
    """带 consultant_agent 的假路由器,记录咨询是否被调用(应不被调用)。"""

    def __init__(self):
        self.consultant_agent = _RecordingConsultant()
        self.recommendation_agent = None
        self.trial_agent = None
        self.lesson_agent = None

    def get_available_services(self):
        return ["咨询服务"]

    async def route_to_consultation(self, task):
        self.consultant_agent.called = True
        yield "[REPLY][归类机器人]\n顾问：没有实时档期数据\n"


class _RecordingConsultant:
    def __init__(self):
        self.called = False


def _make_processor(handler):
    state_manager = MagicMock()
    state_manager.should_classify.return_value = True
    state_manager.is_in_recommendation_confirm_flow.return_value = False
    state_manager.is_in_trial_confirm_flow.return_value = False
    router = _FakeRouter()
    processor = ClassificationProcessor(
        task_classifier=_AsyncConsultationClassifier(),
        state_manager=state_manager,
        agent_router=router,
        unrelated_handler=MagicMock(spec=UnrelatedHandler),
        availability_handler=handler,
    )
    return processor, router


class TestClassificationAvailabilityIntercept:

    def test_stream_intercepts_before_consultation(self):
        """'张明老师什么时候有空' 即使分类器判 consultation,也先走档期查询。"""
        processor, router = _make_processor(_processed_handler(conflict=False))
        out = _collect(processor.process_task_stream("张明老师什么时候有空"))
        assert "可用时段" in out or "可预约" in out
        assert "没有实时档期" not in out
        assert router.consultant_agent.called is False

    def test_sync_intercepts_before_consultation(self):
        """同步版同样拦截,返回完整时段消息串。"""
        processor, router = _make_processor(_processed_handler(conflict=False))
        out = _run(processor.process_task_sync("张明老师什么时候有空"))
        assert "可预约" in out
        assert "没有实时档期" not in out
        assert router.consultant_agent.called is False

    def test_unknown_teacher_from_processor_guides(self):
        """未知老师 → 引导消息,不落咨询。"""
        processor, router = _make_processor(_processed_handler())
        out = _collect(processor.process_task_stream("王五老师什么时候有空"))
        assert "请告诉我要查询哪位老师" in out
        assert router.consultant_agent.called is False


# ----------------------------------------------------------------------
# 5. 非档期输入不受影响(仍走分类)
# ----------------------------------------------------------------------

class TestNonAvailabilityUnaffected:

    def test_normal_consultation_still_routes(self):
        """非档期词命中的咨询输入,不被拦截,正常走咨询路由。"""
        processor, router = _make_processor(_processed_handler())
        out = _collect(processor.process_task_stream("初二孩子数学怎么提高"))
        assert router.consultant_agent.called is True
        assert "没有实时档期数据" in out