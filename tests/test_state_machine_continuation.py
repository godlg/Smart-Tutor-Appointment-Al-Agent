"""
状态机续接测试：试讲/正式课多轮流程不再被重新分类

背景:家长『试讲 → 顾问问时间 → 答“周二晚上6点”』时,旧实现每轮都走 LLM 分类器,
 "周二晚上6点" 被重新归为 lesson → 正式课助手从头问学科/年级。

修复(选项 B,路由层写状态):
- StateManager 局部 FlowState 新增 TRIAL / LESSON 与对应 transition/is_* 方法
- ClassificationProcessor 在 category=="trial"/"lesson" 时先 transition 再路由
- AgentRouter.route_by_state 对 trial/lesson 态直接续接对应 agent(不带首轮分流文案)
- 流程结束/无关走既有 _reset_state_after_* / unrelated 分支把 state 置回 CLASSIFY

异步用例统一用 _run(_collect) 同步包装,不依赖 pytest-asyncio(本仓库未安装)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agents.task_classification.agent_router import (  # noqa: E402
    AgentRouter,
)
from agents.task_classification.classification_processor import (  # noqa: E402
    ClassificationProcessor,
)
from agents.task_classification.state_manager import (  # noqa: E402
    FlowState,
    SharedState,
    StateManager,
)
from config.constants import StateEnum  # noqa: E402

TRIAL_FIRST_COPY = "这是一个试讲预约请求"
LESSON_FIRST_COPY = "这是一个正式课预约"


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
# 测试替身
# ----------------------------------------------------------------------

class _FakeClassifier:
    """固定返回某分类结果,并记录被调次数(判断是否重新分类)。"""

    def __init__(self, category):
        self.category = category
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        return self.category


class _FakeTrialAgent:
    """记录续接调用的试讲假替身。"""

    def __init__(self):
        self.calls = []
        self.shared_state = None

    def set_shared_state(self, state):
        self.shared_state = state

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][试讲助手]已收到"


class _FakeLessonAgent:
    """记录续接调用的正式课假替身。"""

    def __init__(self):
        self.calls = []
        self.shared_state = None

    def set_shared_state(self, state):
        self.shared_state = state

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][正式课助手]已收到"


class _FakeResettingTrialAgent:
    """续接后把 state 置回 CLASSIFY 的试讲替身(模拟 _reset_state_after_trial)。"""

    def __init__(self):
        self.shared_state = None

    def set_shared_state(self, state):
        self.shared_state = state

    async def run_stream(self, user_input=None):
        assert self.shared_state is not None
        # 流程完成 → 置回 CLASSIFY(等价 TrialAgent._reset_state_after_trial)
        self.shared_state.value = StateEnum.CLASSIFY
        yield "[REPLY]完成"


# ----------------------------------------------------------------------
# 1. StateManager：trial / lesson 状态定义与流转
# ----------------------------------------------------------------------

class TestStateManagerTrialLesson:

    def test_flow_state_contains_trial_and_lesson(self):
        assert FlowState.TRIAL.value == "trial"
        assert FlowState.LESSON.value == "lesson"

    def test_transition_to_trial_and_is_in_trial_flow(self):
        sm = StateManager()
        assert not sm.is_in_trial_flow()
        sm.transition_to_trial()
        assert sm.is_in_trial_flow()
        assert not sm.should_classify()          # 不再需要分类
        assert sm.get_state_description() == "试讲流程状态"

    def test_transition_to_lesson_and_is_in_lesson_flow(self):
        sm = StateManager()
        assert not sm.is_in_lesson_flow()
        sm.transition_to_lesson()
        assert sm.is_in_lesson_flow()
        assert not sm.should_classify()
        assert sm.get_state_description() == "正式课流程状态"

    def test_force_reset_clears_trial_lesson(self):
        sm = StateManager()
        sm.transition_to_trial()
        sm.force_reset()
        assert sm.should_classify()
        assert not sm.is_in_trial_flow()
        sm.transition_to_lesson()
        sm.force_reset()
        assert sm.should_classify()
        assert not sm.is_in_lesson_flow()

    def test_enum_value_aligned_with_constants(self):
        """局部 FlowState 与 config.StateEnum value 对齐(按值比较等价)。"""
        assert StateEnum.TRIAL == FlowState.TRIAL
        assert StateEnum.LESSON == FlowState.LESSON


# ----------------------------------------------------------------------
# 2. ClassificationProcessor：分类结果 → 写状态 + 路由
# ----------------------------------------------------------------------

class TestClassificationProcessorStateWrites:

    def test_category_trial_sets_state_and_routes(self):
        # Arrange
        trial_agent = _FakeTrialAgent()
        sm = StateManager()
        router = AgentRouter(trial_agent=trial_agent, state_manager=sm)
        processor = ClassificationProcessor(
            task_classifier=_FakeClassifier("trial"),
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        # Act
        _run(_drain(processor.process_task_stream("想试试张老师的课")))

        # Assert：状态置为 trial，且路由到了 TrialAgent
        assert sm.is_in_trial_flow()
        assert trial_agent.calls == ["想试试张老师的课"]

    def test_category_lesson_sets_state_and_routes(self):
        # Arrange
        lesson_agent = _FakeLessonAgent()
        sm = StateManager()
        router = AgentRouter(lesson_agent=lesson_agent, state_manager=sm)
        processor = ClassificationProcessor(
            task_classifier=_FakeClassifier("lesson"),
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        # Act
        _run(_drain(processor.process_task_stream("预约明天下午3点数学课")))

        # Assert
        assert sm.is_in_lesson_flow()
        assert lesson_agent.calls == ["预约明天下午3点数学课"]

    def test_state_trial_second_round_does_not_reclassify(self):
        """state 已为 trial 时不再调 classify_task,直接走续接,且无首轮文案。"""
        # Arrange
        trial_agent = _FakeTrialAgent()
        sm = StateManager()
        router = AgentRouter(trial_agent=trial_agent, state_manager=sm)
        classifier = _FakeClassifier("trial")
        processor = ClassificationProcessor(
            task_classifier=classifier,
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        # 第一轮：分类 → trial
        first = _collect(processor.process_task_stream("想试试张老师的课"))
        assert classifier.calls == 1
        assert TRIAL_FIRST_COPY in first          # 首轮分流文案(预期存在)

        # 第二轮：state=TRIAL,不应再分类
        second = _collect(processor.process_task_stream("周二晚上6点"))
        assert classifier.calls == 1              # 未重新分类
        assert TRIAL_FIRST_COPY not in second     # 续接不带首轮文案
        assert trial_agent.calls[-1] == "周二晚上6点"

    def test_sync_category_trial_sets_state(self):
        # Arrange
        trial_agent = _FakeTrialAgent()
        sm = StateManager()
        router = AgentRouter(trial_agent=trial_agent, state_manager=sm)
        processor = ClassificationProcessor(
            task_classifier=_FakeClassifier("trial"),
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        # Act
        result = _run(processor.process_task_sync("想试试张老师的课"))

        # Assert
        assert sm.is_in_trial_flow()
        assert trial_agent.calls == ["想试试张老师的课"]
        assert result != "暂不支持该类型任务"

    def test_sync_continuation_second_round_no_reclassify(self):
        # Arrange
        trial_agent = _FakeTrialAgent()
        sm = StateManager()
        router = AgentRouter(trial_agent=trial_agent, state_manager=sm)
        classifier = _FakeClassifier("trial")
        processor = ClassificationProcessor(
            task_classifier=classifier,
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )
        _run(processor.process_task_sync("想试试张老师的课"))
        assert classifier.calls == 1

        # Act：第二轮回调续接,不应再分类
        result = _run(processor.process_task_sync("周二晚上6点"))

        # Assert
        assert classifier.calls == 1
        assert TRIAL_FIRST_COPY not in result
        assert result != "暂不支持该类型任务"
        assert trial_agent.calls[-1] == "周二晚上6点"


# ----------------------------------------------------------------------
# 3. route_by_state：trial / lesson 态直接续接缺省 agent(无首轮文案)
# ----------------------------------------------------------------------

class TestRouteByStateContinuation:

    def test_state_trial_routes_to_trial_agent_without_first_copy(self):
        # Arrange
        trial_agent = _FakeTrialAgent()
        lesson_agent = _FakeLessonAgent()
        sm = StateManager()
        sm.transition_to_trial()
        router = AgentRouter(
            trial_agent=trial_agent, lesson_agent=lesson_agent, state_manager=sm
        )

        # Act
        output = _collect(router.route_by_state("周二晚上6点"))

        # Assert：回试讲,且不带首轮分流文案
        assert trial_agent.calls == ["周二晚上6点"]
        assert lesson_agent.calls == []
        assert TRIAL_FIRST_COPY not in output

    def test_state_lesson_routes_to_lesson_agent_without_first_copy(self):
        # Arrange
        trial_agent = _FakeTrialAgent()
        lesson_agent = _FakeLessonAgent()
        sm = StateManager()
        sm.transition_to_lesson()
        router = AgentRouter(
            trial_agent=trial_agent, lesson_agent=lesson_agent, state_manager=sm
        )

        # Act
        output = _collect(router.route_by_state("时间是周三下午"))

        # Assert：回正式课,不带首轮分流文案
        assert lesson_agent.calls == ["时间是周三下午"]
        assert trial_agent.calls == []
        assert LESSON_FIRST_COPY not in output

    def test_state_trial_agent_error_is_surfaced(self):
        # Arrange
        class _BoomTrialAgent:
            def set_shared_state(self, state):
                pass

            async def run_stream(self, user_input=None):
                yield "[INCOMPLETE]"
                raise RuntimeError("boom")

        sm = StateManager()
        sm.transition_to_trial()
        router = AgentRouter(trial_agent=_BoomTrialAgent(), state_manager=sm)

        # Act
        output = _collect(router.route_by_state("周二晚上6点"))

        # Assert：错误显式回报而非静默吞掉
        assert "[ERROR]试讲处理失败" in output


# ----------------------------------------------------------------------
# 4. 回归：TrialAgent unrelated 仍置 CLASSIFY + 递归防护仍在(配合新状态机共享状态)
# ----------------------------------------------------------------------

class TestTrialUnrelatedStillResetsState:

    @staticmethod
    def _make_agent(state):
        """绕过 __init__(避免真实 LLM),构造可跑 unrelated 分支的 TrialAgent。"""
        from agents.trial.trial_agent import TrialAgent

        agent = object.__new__(TrialAgent)
        agent.processor = MagicMock()
        agent.unrelated_callback = None
        agent.state = state
        agent.trial_history = {
            "subject": None, "grade_level": None, "start_time": None,
            "duration_minutes": "30", "student_situation": None,
            "preferred_teacher_name": None, "preferred_style": None,
            "unrelated_escalations": 0,
        }
        agent.chat_history = None
        agent.session_id = "test-session"

        entry = {
            "subject": "未知", "grade_level": "未知", "start_time": "未知",
            "duration_minutes": "30", "student_situation": "",
            "preferred_teacher_name": "未知", "preferred_style": "未知",
            "confirmation": "未知", "info_complete": False, "unrelated": True,
            "missing_info": ["所有信息"],
        }
        agent.processor.input_parser.parse_stream.return_value = ['{"unrelated": true}']
        agent.processor.input_parser.parse_data.return_value = entry
        agent.processor.update_history_from_data.return_value = False

        calls = []

        async def _unrelated_gen(user_input, unrelated_callback, state):
            if unrelated_callback is not None:
                calls.append(user_input)
            yield "[UNRELATED]"

        agent.processor.handle_unrelated_request = MagicMock(side_effect=_unrelated_gen)
        # unrelated=true 提前 return,不会走到 handle_incomplete_info
        agent.processor.handle_incomplete_info = MagicMock()
        return agent

    def test_unrelated_in_trial_flow_resets_to_classify(self):
        # state 与本机状态管理器共享同一 SharedState,初始为 TRIAL
        shared = SharedState()
        shared.value = StateEnum.TRIAL
        agent = self._make_agent(shared)

        out = _collect(agent.run_stream(user_input="今天天气怎么样"))

        assert "[UNRELATED]" in out
        # unrelated 踢回 → state 被置回 CLASSIFY
        assert shared.value == StateEnum.CLASSIFY
        # 递归防护计数照旧(第一次 = 1)
        assert agent.trial_history["unrelated_escalations"] == 1


# ----------------------------------------------------------------------
# 5. 端到端续接(轻量)：答时间路由回 trial,而非 lesson,且完成态回到 CLASSIFY
# ----------------------------------------------------------------------

class TestEndToEndContinuation:

    def test_answer_time_routes_back_to_trial_not_lesson(self):
        """state=TRIAL 时,`周二晚上6点` 必须回试讲,而不是跳到正式课。"""
        trial_agent = _FakeTrialAgent()
        lesson_agent = _FakeLessonAgent()
        sm = StateManager()
        sm.transition_to_trial()
        router = AgentRouter(
            trial_agent=trial_agent, lesson_agent=lesson_agent, state_manager=sm
        )
        processor = ClassificationProcessor(
            task_classifier=_FakeClassifier("trial"),
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        out = _collect(processor.process_task_stream("周二晚上6点"))

        assert trial_agent.calls == ["周二晚上6点"]
        assert lesson_agent.calls == []            # 没有跳到正式课
        assert TRIAL_FIRST_COPY not in out

    def test_completed_flow_resets_state_to_classify(self):
        """续接完成(替身模拟 _reset_state_after_trial)后回到 CLASSIFY。"""
        trial_agent = _FakeResettingTrialAgent()
        sm = StateManager()
        sm.transition_to_trial()
        router = AgentRouter(trial_agent=trial_agent, state_manager=sm)
        processor = ClassificationProcessor(
            task_classifier=_FakeClassifier("trial"),
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(),
        )

        _collect(processor.process_task_stream("是"))

        assert sm.should_classify()
        assert not sm.is_in_trial_flow()