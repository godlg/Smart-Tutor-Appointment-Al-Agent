"""
推荐确认态跳过分类器直接转 RecommendationAgent 测试

覆盖 Bug 1:家长在推荐确认态下回『张明/ID 7/是/不』时,主路由仍走 LLM 分类器
→ 把它判成 recommendation_confirm → 路由回推荐 agent,但前置一段
『[归类机器人] 这是一个老师推荐请求』错乱文案。

修复:
1. StateEnum 新增 RECOMMENDATION_CONFIRM 状态
2. RecommendationAgent 进入 awaiting_confirmation=True 时立即把 state 置 RECOMMENDATION_CONFIRM
3. ClassificationProcessor.process_task_stream / process_task_sync 优先判断
   state_manager.is_in_recommendation_confirm_flow(),为 True 时跳过分类器直接转推荐 agent
4. AgentRouter.route_to_recommendation_confirm 跳过首轮分流文案

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import StateEnum  # noqa: E402
from agents.recommendation_agent import RecommendationAgent  # noqa: E402
from agents.task_classification.agent_router import AgentRouter  # noqa: E402
from agents.task_classification.classification_processor import ClassificationProcessor  # noqa: E402
from agents.task_classification.state_manager import StateManager  # noqa: E402
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
# 1. StateManager:新增 RECOMMENDATION_CONFIRM 转换 + 判定
# ----------------------------------------------------------------------

class TestStateManagerRecommendationConfirm:
    """StateManager 必须支持 RECOMMENDATION_CONFIRM 状态转换与判定"""

    def test_default_is_classify(self):
        sm = StateManager()
        assert sm.get_current_state() == StateEnum.CLASSIFY
        assert sm.is_in_recommendation_confirm_flow() is False

    def test_transition_to_recommendation_confirm(self):
        sm = StateManager()
        sm.transition_to_recommendation_confirm()
        assert sm.get_current_state() == StateEnum.RECOMMENDATION_CONFIRM
        assert sm.is_in_recommendation_confirm_flow() is True
        # transition 后 should_classify=False(下一轮不再走分类器)
        assert sm.should_classify() is False

    def test_state_description_includes_recommendation_confirm(self):
        sm = StateManager()
        sm.transition_to_recommendation_confirm()
        assert sm.get_state_description() == "推荐确认态"


# ----------------------------------------------------------------------
# 2. AgentRouter:route_to_recommendation_confirm 跳过首轮分流文案
# ----------------------------------------------------------------------

class _FakeRecommendationAgent:
    """假 RecommendationAgent:记录被续接的输入,并给出回复"""

    def __init__(self):
        self.calls = []
        self.did_yield_thought = False

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][推荐助手]推荐确认续轮结果"


class TestRouteToRecommendationConfirm:
    """route_to_recommendation_confirm 不 yield 归类机器人首轮分流文案"""

    def test_route_skips_classification_thought(self):
        agent = _FakeRecommendationAgent()
        router = AgentRouter(recommendation_agent=agent, state_manager=MagicMock())
        out = _run(_drain(router.route_to_recommendation_confirm("张明")))
        # 关键:不应出现「[THOUGHT][归类机器人] 这是一个老师推荐请求」
        assert "[THOUGHT][归类机器人] 这是一个老师推荐请求" not in out
        assert "推荐确认续轮结果" in out
        assert agent.calls == ["张明"]

    def test_route_without_agent_yields_error(self):
        router = AgentRouter(recommendation_agent=None, state_manager=MagicMock())
        out = _run(_drain(router.route_to_recommendation_confirm("任意输入")))
        assert "[ERROR]" in out

    def test_route_differs_from_route_to_recommendation(self):
        """与 route_to_recommendation 对比:后者有分流文案,前者没有"""
        agent = _FakeRecommendationAgent()
        router = AgentRouter(recommendation_agent=agent, state_manager=MagicMock())
        out1 = _run(_drain(router.route_to_recommendation("任意输入")))
        out2 = _run(_drain(router.route_to_recommendation_confirm("任意输入")))
        # 老路由有分流文案
        assert "[THOUGHT][归类机器人] 这是一个老师推荐请求" in out1
        # 新路由不带
        assert "[THOUGHT][归类机器人]" not in out2


# ----------------------------------------------------------------------
# 3. ClassificationProcessor:推荐确认态优先,跳过分类器
# ----------------------------------------------------------------------

class _CounterClassifier:
    """记录调用次数的分类器(用于断言『第 N 轮不应被分类』)"""

    def __init__(self, return_category="recommendation_confirm"):
        self.call_count = 0
        self.return_category = return_category

    async def classify_task(self, task):
        self.call_count += 1
        return self.return_category


def _make_recommendation_agent():
    """绕过 __init__(避免真实 LLM)构造可跑确认状态机的 RecommendationAgent。"""
    agent = RecommendationAgent.__new__(RecommendationAgent)
    agent.session_id = "s-rc"
    agent.chats_by_session_id = {}
    agent.history = {}
    agent.message_builder = RecommendationAgent(session_id="b").message_builder
    agent.input_parser = MagicMock()
    agent._recommendation_service = MagicMock()
    agent._student_service = MagicMock()
    # __new__ 缺 state,显式给一个共享状态
    agent.state = MagicMock()
    return agent


class TestClassificationProcessorRecommendationConfirmBypass:
    """主路由在 RECOMMENDATION_CONFIRM 状态时跳过分类器"""

    def test_stream_skips_classifier_in_confirm_state(self):
        """state=RECOMMENDATION_CONFIRM → process_task_stream 跳过分类器直接进推荐 agent"""
        rec = _make_recommendation_agent()
        sm = StateManager()
        # 模拟推荐 agent 已进入确认态(直接置 state)
        sm.transition_to_recommendation_confirm()
        # 用真 router,但 classification_processor 内部会调 agent_router.route_to_recommendation_confirm
        # 我们用一个 spy router 来观察
        from agents.task_classification.agent_router import AgentRouter

        router = AgentRouter(
            consultant_agent=None,
            state_manager=sm,
            trial_agent=None,
            lesson_agent=None,
            recommendation_agent=rec,
        )
        classifier = _CounterClassifier()
        processor = ClassificationProcessor(
            task_classifier=classifier,
            state_manager=sm,
            agent_router=router,
            unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        prior_calls = classifier.call_count
        out = _collect(processor.process_task_stream("张明"))
        # 关键:分类器没被调
        assert classifier.call_count == prior_calls
        # 输出不含归类机器人首轮分流文案
        assert "[THOUGHT][归类机器人] 这是一个老师推荐请求" not in out

    def test_sync_skips_classifier_in_confirm_state(self):
        sm = StateManager()
        sm.transition_to_recommendation_confirm()
        rec = _make_recommendation_agent()
        router = AgentRouter(
            consultant_agent=None, state_manager=sm, trial_agent=None,
            lesson_agent=None, recommendation_agent=rec,
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

    def test_classifier_still_called_in_normal_state(self):
        """state=CLASSIFY(默认)时,分类器照常被调,不影响老路径"""
        rec = _make_recommendation_agent()
        sm = StateManager()
        # 显式 reset 回 CLASSIFY(默认状态,这里只是显式表达)
        assert sm.get_current_state() == StateEnum.CLASSIFY
        router = AgentRouter(
            consultant_agent=None, state_manager=sm, trial_agent=None,
            lesson_agent=None, recommendation_agent=rec,
        )
        classifier = _CounterClassifier(return_category="recommendation")
        processor = ClassificationProcessor(
            task_classifier=classifier, state_manager=sm,
            agent_router=router, unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        prior_calls = classifier.call_count
        out = _collect(processor.process_task_stream("请推荐一位老师"))
        assert classifier.call_count == prior_calls + 1  # 分类器被调
        # CLASSIFY 状态走老路由,有分流文案
        assert "[THOUGHT][归类机器人] 这是一个老师推荐请求" in out


# ----------------------------------------------------------------------
# 4. 端到端集成:4 轮真实流程——每轮分类器调用次数严格符合预期
# ----------------------------------------------------------------------

class TestEndToEndClassificationBypass:
    """完整 4 轮:推荐→选老师→确认『是』→报时间,只在第 1 轮走分类器"""

    def test_4_turn_classifier_called_once(self):
        rec = _make_recommendation_agent()
        rec.input_parser.extract.return_value = {
            "subject": "数学", "grade_level": "初中", "preferred_style": "未知",
            "preferred_teacher_name": "未知", "mode": "未知",
            "info_complete": True, "missing_info": [],
        }
        rec._recommendation_service.recommend_teachers.return_value = [
            {"teacher": {"id": 7, "name": "张明"}, "score": 92, "reason": "擅长初二数学"},
        ]
        rec._student_service.initialize_default_student.return_value = {"id": 1}

        sm = StateManager()
        router = AgentRouter(
            consultant_agent=None, state_manager=sm, trial_agent=None,
            lesson_agent=None, recommendation_agent=rec,
        )
        classifier = _CounterClassifier()
        processor = ClassificationProcessor(
            task_classifier=classifier, state_manager=sm,
            agent_router=router, unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )

        # 第 1 轮:推荐 → 分类器被调 → state 置 RECOMMENDATION_CONFIRM(由 recommendation_agent.run_stream 内置)
        _collect(processor.process_task_stream("初二数学"))
        assert classifier.call_count == 1
        assert sm.is_in_recommendation_confirm_flow() is True

        # 第 2 轮:回『张明』→ 跳过分类器
        prior = classifier.call_count
        _collect(processor.process_task_stream("张明"))
        assert classifier.call_count == prior

        # 第 3 轮:回『是』→ 跳过分类器
        prior = classifier.call_count
        _collect(processor.process_task_stream("是"))
        assert classifier.call_count == prior
