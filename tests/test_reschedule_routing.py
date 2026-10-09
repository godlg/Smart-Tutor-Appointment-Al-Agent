"""
调课(reschedule) / 退课(cancel) 聊天入口路由测试

覆盖修复前的缺口：
- 分类结果为 reschedule / cancel 时会落入 else → handle_unsupported_task 拒答
- AgentRouter 没有调课/退课路由方法
- LessonAgent 不区分意图，调课请求会被当成「预约一节新课」

测试全部 mock 掉 LLM 与 services，不发起外部 API 调用、不触碰生产数据库。
异步用例统一走 _run(asyncio.run) 包装，与 tests/test_consultant_agent.py 保持一致。
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from agents.lesson.change_processor import (
    LessonChangeMessageBuilder,
    LessonChangeParser,
    LessonChangeProcessor,
)
from agents.lesson.reschedule_handler import RescheduleMode
from agents.task_classification.agent_router import AgentRouter
from agents.task_classification.classification_processor import ClassificationProcessor

UNSUPPORTED_MARKER = "超出我的能力范围"


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream) -> str:
    """把异步 token 流拼成完整回复"""
    tokens = []
    async for token in stream:
        tokens.append(token)
    return "".join(tokens)


async def _empty_stream():
    """空 token 流，用于替换 handle_unsupported_task 做调用探针"""
    return
    yield  # pragma: no cover


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------

class _FakeLessonAgent:
    """记录被调用时收到的 user_input / intent"""

    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None, intent="lesson"):
        self.calls.append({"user_input": user_input, "intent": intent})
        yield f"[REPLY][正式课助手]已按 {intent} 处理"


class _LegacyLessonAgent:
    """不认识 intent 参数的旧版 agent，用于验证异常被显式回报而非静默吞掉"""

    async def run_stream(self, user_input=None):
        yield "[REPLY]legacy"


class _FakeClassifier:
    """固定返回某个分类结果的分类器"""

    def __init__(self, category):
        self.category = category

    async def classify_task(self, task):
        return self.category


class _FakeStateManager:
    """始终处于「可分类」状态"""

    def __init__(self):
        self.state = None
        self.reset_called = False

    def should_classify(self):
        return True

    def force_reset(self):
        self.reset_called = True

    def transition_to_trial(self):
        """对齐 StateManager 接口(分类流程会在 category==trial 时调用)"""
        pass

    def transition_to_lesson(self):
        """对齐 StateManager 接口(分类流程会在 category==lesson 时调用)"""
        pass

    def is_in_recommendation_confirm_flow(self):
        """对齐 StateManager 接口(主路由推荐确认态优先分支会调用)"""
        return False

    def is_in_trial_confirm_flow(self):
        """对齐 StateManager 接口(主路由试讲预约确认态优先分支会调用)"""
        return False

    def is_in_trial_convert_flow(self):
        """对齐 StateManager 接口(主路由试讲转化态优先分支会调用)"""
        return False


def _build_processor(category, lesson_agent=None):
    """组装一个只挂了 lesson_agent 的分类流程处理器"""
    router = AgentRouter(
        lesson_agent=lesson_agent if lesson_agent is not None else _FakeLessonAgent(),
        state_manager=_FakeStateManager(),
    )
    processor = ClassificationProcessor(
        task_classifier=_FakeClassifier(category),
        state_manager=_FakeStateManager(),
        agent_router=router,
        unrelated_handler=MagicMock(),
    )
    return processor, router


# ----------------------------------------------------------------------
# 1. 分类流程：reschedule / cancel 不再落入 handle_unsupported_task
# ----------------------------------------------------------------------

class TestClassificationRouting:

    @pytest.mark.parametrize("category", ["reschedule", "cancel"])
    def test_stream_routes_change_intent_to_lesson_agent(self, category):
        # Arrange
        agent = _FakeLessonAgent()
        processor, _ = _build_processor(category, lesson_agent=agent)

        # Act
        output = _run(_collect(processor.process_task_stream("把周三的课改到周五")))

        # Assert：真正走到 LessonAgent，且带上正确意图
        assert agent.calls == [{"user_input": "把周三的课改到周五", "intent": category}]
        assert UNSUPPORTED_MARKER not in output

    @pytest.mark.parametrize("category", ["reschedule", "cancel"])
    def test_stream_does_not_call_unsupported_handler(self, category, monkeypatch):
        # Arrange
        processor, router = _build_processor(category)
        called = []
        monkeypatch.setattr(
            router, "handle_unsupported_task",
            lambda c: called.append(c) or _empty_stream(),
        )

        # Act
        _run(_collect(processor.process_task_stream("帮我取消明天的课")))

        # Assert
        assert called == []

    @pytest.mark.parametrize("category", ["reschedule", "cancel"])
    def test_sync_routes_change_intent_to_lesson_agent(self, category):
        # Arrange
        agent = _FakeLessonAgent()
        processor, _ = _build_processor(category, lesson_agent=agent)

        # Act
        result = _run(processor.process_task_sync("帮我取消明天的课"))

        # Assert
        assert agent.calls[0]["intent"] == category
        assert result != "暂不支持该类型任务"

    def test_unknown_category_still_falls_back_to_unsupported(self):
        """回归保护：其它未知意图仍应走拒答分支"""
        # Arrange
        processor, _ = _build_processor("something_else")

        # Act
        output = _run(_collect(processor.process_task_stream("今天天气怎么样")))

        # Assert
        assert UNSUPPORTED_MARKER in output

    @pytest.mark.parametrize("category", ["reschedule", "cancel"])
    def test_change_intent_without_lesson_agent_falls_back(self, category):
        """没有 lesson_agent 时不应崩溃，退回拒答"""
        # Arrange
        processor, router = _build_processor(category)
        router.lesson_agent = None

        # Act
        output = _run(_collect(processor.process_task_stream("把周三的课改到周五")))

        # Assert
        assert UNSUPPORTED_MARKER in output

    @pytest.mark.parametrize("category", ["trial", "lesson", "consultation"])
    def test_existing_routes_are_untouched(self, category):
        """回归保护：原有分支的意图不受影响"""
        # Arrange
        agent = _FakeLessonAgent()
        processor, router = _build_processor(category, lesson_agent=agent)

        # Act
        _run(_collect(processor.process_task_stream("给我孩子预约明天下午3点的数学课")))

        # Assert：只有 lesson 会落到 lesson_agent，且仍是默认预约意图
        if category == "lesson":
            assert agent.calls[0]["intent"] == "lesson"
        else:
            assert agent.calls == []


# ----------------------------------------------------------------------
# 2. AgentRouter：新增的调课/退课路由方法
# ----------------------------------------------------------------------

class TestAgentRouterChangeRoutes:

    def test_route_to_reschedule_passes_intent(self):
        # Arrange
        agent = _FakeLessonAgent()
        router = AgentRouter(lesson_agent=agent, state_manager=_FakeStateManager())

        # Act
        output = _run(_collect(router.route_to_reschedule("把周三的课改到周五")))

        # Assert
        assert agent.calls[0]["intent"] == "reschedule"
        assert "调课" in output

    def test_route_to_cancel_passes_intent(self):
        # Arrange
        agent = _FakeLessonAgent()
        router = AgentRouter(lesson_agent=agent, state_manager=_FakeStateManager())

        # Act
        output = _run(_collect(router.route_to_cancel("帮我取消明天的课")))

        # Assert
        assert agent.calls[0]["intent"] == "cancel"
        assert "退课" in output

    @pytest.mark.parametrize("method", ["route_to_reschedule", "route_to_cancel"])
    def test_change_route_without_agent_yields_error(self, method):
        # Arrange
        router = AgentRouter(lesson_agent=None, state_manager=_FakeStateManager())

        # Act
        output = _run(_collect(getattr(router, method)("把周三的课改到周五")))

        # Assert
        assert "[ERROR]" in output

    def test_agent_failure_is_surfaced_not_swallowed(self):
        """旧版 agent 不接受 intent 时，错误应显式回报给家长"""
        # Arrange
        router = AgentRouter(
            lesson_agent=_LegacyLessonAgent(), state_manager=_FakeStateManager()
        )

        # Act
        output = _run(_collect(router.route_to_reschedule("把周三的课改到周五")))

        # Assert
        assert "[ERROR]调课处理失败" in output

    def test_available_services_include_change_capability(self):
        # Arrange
        router = AgentRouter(
            lesson_agent=_FakeLessonAgent(), state_manager=_FakeStateManager()
        )

        # Act
        services = router.get_available_services()

        # Assert
        assert "调课/退课" in services


# ----------------------------------------------------------------------
# 3. LessonChangeProcessor：自然语言 → lesson_id → 复用已有 Handler
# ----------------------------------------------------------------------

def _build_change_processor(lessons, parsed_data):
    """构造一个不碰 LLM、不碰数据库的调课/退课处理器

    用 __new__ 跳过 __init__，避免真实创建 LLM 与 services（否则会连生产 DB）。
    """

    class _FakeParser:
        async def parse(self, user_input, lessons_text, intent):
            return parsed_data

    processor = LessonChangeProcessor.__new__(LessonChangeProcessor)
    processor.llm = MagicMock()
    processor.parser = _FakeParser()
    processor.message_builder = LessonChangeMessageBuilder()
    processor.reschedule_handler = MagicMock()
    processor.cancel_handler = MagicMock()

    processor.student_service = MagicMock()
    processor.student_service.initialize_default_student.return_value = {"id": 1}
    processor.lesson_service = MagicMock()
    processor.lesson_service.get_student_upcoming_lessons.return_value = lessons
    processor.teacher_service = MagicMock()
    processor.teacher_service.get_teacher_by_id.return_value = {"name": "张老师"}
    return processor


def _make_lesson(lesson_id=101, days_ahead=2, series_id=None):
    return {
        "id": lesson_id,
        "teacher_id": 7,
        "student_id": 1,
        "series_id": series_id,
        "subject": "数学",
        "scheduled_at": datetime.now() + timedelta(days=days_ahead),
        "duration_minutes": 60,
        "status": "confirmed",
        "price": 200.0,
    }


class TestLessonChangeProcessor:

    def test_no_upcoming_lessons_returns_friendly_message(self):
        # Arrange
        processor = _build_change_processor(lessons=[], parsed_data={})

        # Act
        output = _run(_collect(
            processor.handle_change_request("把周三的课改到周五", "reschedule")
        ))

        # Assert
        assert "没有待上的课时" in output
        processor.reschedule_handler.reschedule.assert_not_called()

    def test_reschedule_calls_handler_with_parsed_time(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson()],
            parsed_data={
                "target_lesson_id": 101,
                "new_time": "2030-06-07 16:00",
                "scope": RescheduleMode.ONLY_THIS,
                "reason": "未知",
            },
        )
        processor.reschedule_handler.reschedule.return_value = (True, "调课成功")

        # Act
        output = _run(_collect(
            processor.handle_change_request("把周三的课改到周五下午4点", "reschedule")
        ))

        # Assert
        processor.reschedule_handler.reschedule.assert_called_once_with(
            lesson_id=101,
            new_time=datetime(2030, 6, 7, 16, 0, tzinfo=timezone(timedelta(hours=8))),
            mode=RescheduleMode.ONLY_THIS,
        )
        assert "调课成功" in output

    def test_reschedule_cascade_scope_is_passed_through(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson(series_id=9)],
            parsed_data={
                "target_lesson_id": 101,
                "new_time": "2030-06-07 16:00",
                "scope": RescheduleMode.THIS_AND_FOLLOWING,
                "reason": "未知",
            },
        )
        processor.reschedule_handler.reschedule.return_value = (True, "级联调课成功")

        # Act
        _run(_collect(
            processor.handle_change_request("以后每周都改到周五", "reschedule")
        ))

        # Assert
        assert (
            processor.reschedule_handler.reschedule.call_args.kwargs["mode"]
            == RescheduleMode.THIS_AND_FOLLOWING
        )

    def test_reschedule_failure_is_reported(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson()],
            parsed_data={
                "target_lesson_id": 101,
                "new_time": "2030-06-07 16:00",
                "scope": RescheduleMode.ONLY_THIS,
                "reason": "未知",
            },
        )
        processor.reschedule_handler.reschedule.return_value = (
            False, "新时间冲突：老师已被预约",
        )

        # Act
        output = _run(_collect(
            processor.handle_change_request("改到周五", "reschedule")
        ))

        # Assert
        assert "没有成功" in output
        assert "老师已被预约" in output

    def test_reschedule_without_new_time_asks_for_it(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson()],
            parsed_data={
                "target_lesson_id": 101,
                "new_time": "未知",
                "scope": RescheduleMode.ONLY_THIS,
                "reason": "未知",
            },
        )

        # Act
        output = _run(_collect(
            processor.handle_change_request("这节课要改一下", "reschedule")
        ))

        # Assert
        assert "希望改到什么时间" in output
        processor.reschedule_handler.reschedule.assert_not_called()

    def test_cancel_calls_handler_and_shows_refund(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson()],
            parsed_data={
                "target_lesson_id": 101,
                "new_time": "未知",
                "scope": RescheduleMode.ONLY_THIS,
                "reason": "孩子生病",
            },
        )
        processor.cancel_handler.cancel_lesson.return_value = (
            True, "退课成功",
            {"original_price": 200.0, "refund_amount": 160.0,
             "refund_ratio": 0.8, "policy": "提前 24h 以上退费 80%"},
        )

        # Act
        output = _run(_collect(
            processor.handle_change_request("帮我取消明天的课", "cancel")
        ))

        # Assert
        processor.cancel_handler.cancel_lesson.assert_called_once_with(
            lesson_id=101, reason="孩子生病"
        )
        assert "退课已办理" in output
        assert "160 元" in output

    def test_cancel_failure_is_reported(self):
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson()],
            parsed_data={"target_lesson_id": 101, "reason": "未知"},
        )
        processor.cancel_handler.cancel_lesson.return_value = (
            False, "已完成的课时无法取消", None,
        )

        # Act
        output = _run(_collect(processor.handle_change_request("取消这节课", "cancel")))

        # Assert
        assert "已完成的课时无法取消" in output

    @pytest.mark.parametrize("bad_id", ["未知", None, 999, "abc"])
    def test_unresolvable_target_never_touches_handlers(self, bad_id):
        """定位不到课时就澄清——绝不猜一节课去改/取消"""
        # Arrange
        processor = _build_change_processor(
            lessons=[_make_lesson(101), _make_lesson(102, days_ahead=4)],
            parsed_data={"target_lesson_id": bad_id, "new_time": "2030-06-07 16:00"},
        )

        # Act
        output = _run(_collect(processor.handle_change_request("取消那节课", "cancel")))

        # Assert
        assert "没能确定" in output
        assert "ID 101" in output  # 列出候选课时供家长选择
        processor.cancel_handler.cancel_lesson.assert_not_called()
        processor.reschedule_handler.reschedule.assert_not_called()

    def test_service_failure_degrades_to_no_lesson_message(self):
        """services 查询异常时不应把异常栈抛给家长"""
        # Arrange
        processor = _build_change_processor(lessons=[], parsed_data={})
        processor.lesson_service.get_student_upcoming_lessons.side_effect = RuntimeError(
            "db down"
        )

        # Act
        output = _run(_collect(processor.handle_change_request("取消明天的课", "cancel")))

        # Assert
        assert "没有待上的课时" in output


# ----------------------------------------------------------------------
# 4. 解析器容错
# ----------------------------------------------------------------------

class TestLessonChangeParser:

    @pytest.mark.parametrize("bad", ["", "not json", "[]", "```json{}```"])
    def test_parse_data_falls_back_to_unknown(self, bad):
        # Arrange
        parser = LessonChangeParser.__new__(LessonChangeParser)

        # Act
        data = parser.parse_data(bad)

        # Assert
        assert data["target_lesson_id"] == "未知"
        assert data["new_time"] == "未知"

    def test_parse_data_reads_valid_json(self):
        # Arrange
        parser = LessonChangeParser.__new__(LessonChangeParser)

        # Act
        data = parser.parse_data(
            '{"target_lesson_id": 12, "new_time": "2030-01-01 10:00"}'
        )

        # Assert
        assert data["target_lesson_id"] == 12
