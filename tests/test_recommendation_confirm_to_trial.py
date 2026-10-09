"""
推荐确认『是』→ 直接进入试讲流程的集成测试

背景:家长对"确认预约试讲请回复『是』"回复『是』后,推荐助手应直接预约试讲
(问时间),不再重复问"回复『试讲』/『正式课』",并把共享状态置 TRIAL,
让下一轮(家长答时间)由状态机直接续接 TrialAgent,而不重新分类。

异步用例统一用 _run(_collect) 同步包装(本仓库未装 pytest-asyncio)。
"""
import asyncio
from unittest.mock import MagicMock

from agents.recommendation_agent import RecommendationAgent
from agents.task_classification.agent_router import AgentRouter
from agents.task_classification.classification_processor import ClassificationProcessor
from agents.task_classification.state_manager import StateManager
from agents.task_classification.unrelated_handler import UnrelatedHandler
from config.constants import StateEnum

_RECS = [
    {"teacher": {"id": 7, "name": "高远"}, "score": 92, "reason": "擅长初二数学"},
    {"teacher": {"id": 9, "name": "张明"}, "score": 80, "reason": "覆盖初中"},
]
_COMPLETE_INPUT = {
    "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
    "preferred_teacher_name": "未知", "mode": "未知",
    "info_complete": True, "missing_info": [],
}


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream):
    tokens = []
    async for tok in stream:
        tokens.append(tok)
    return "".join(tokens)


class _AsyncFakeClassifier:
    """固定返回某类别的异步分类器,并记录调用次数(断言第 4 轮不再分类)。"""

    def __init__(self, category):
        self.category = category
        self.call_count = 0

    async def classify_task(self, task):
        self.call_count += 1
        return self.category


class _FakeTrial:
    """伪 TrialAgent:记录被续接的输入,并给出试讲风格输出。"""

    def __init__(self):
        self.calls = []

    def set_shared_state(self, shared_state):
        self.state = shared_state

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][试讲顾问]为您推荐以下老师，我们预约周二晚上6点试讲。"


def _make_recommendation_agent():
    """构造可跑确认状态机的 RecommendationAgent(绕过 __init__ 的真实 LLM)。"""
    agent = RecommendationAgent.__new__(RecommendationAgent)
    agent.session_id = "s-integ"
    agent.chats_by_session_id = {}
    agent.history = {}
    agent.message_builder = RecommendationAgent(session_id="b").message_builder
    agent.input_parser = MagicMock()
    agent._recommendation_service = MagicMock()
    agent._student_service = MagicMock()
    return agent


class TestRecommendationConfirmToTrial:

    def _make_processor(self, classifier_category="recommendation"):
        rec = _make_recommendation_agent()
        trial = _FakeTrial()
        state_manager = StateManager()
        router = AgentRouter(
            consultant_agent=None,
            state_manager=state_manager,
            trial_agent=trial,
            lesson_agent=None,
            recommendation_agent=rec,
        )
        # AgentRouter.__init__ 已 _setup_agent_states:rec.state = state_manager.state,
        # 确认试讲时改的是这颗共享状态,state_manager 同见。
        assert rec.state is state_manager.state
        classifier = _AsyncFakeClassifier(classifier_category)
        unrelated_handler = MagicMock(spec=UnrelatedHandler)
        processor = ClassificationProcessor(
            task_classifier=classifier,
            state_manager=state_manager,
            agent_router=router,
            unrelated_handler=unrelated_handler,
        )
        return rec, trial, state_manager, classifier, processor

    def test_confirm_yes_sets_shared_state_trial(self):
        """确认『是』后共享状态 immediate 变成 TRIAL(通过真实 set_shared_state 接线)。"""
        rec, _, state_manager, _, _ = self._make_processor()
        # 一轮完整推荐进入确认态
        rec.input_parser.extract.return_value = _COMPLETE_INPUT
        rec._recommendation_service.recommend_teachers.return_value = _RECS
        rec._student_service.initialize_default_student.return_value = {"id": 1}
        _run(_collect(rec.run_stream("初二数学", session_id="s-integ")))
        # 选定老师
        _run(_collect(rec.run_stream("高远", session_id="s-integ")))
        # 确认『是』
        out = _run(_collect(rec.run_stream("是", session_id="s-integ")))

        assert "什么时候试讲" in out
        assert rec.state.value == StateEnum.TRIAL
        assert state_manager.get_current_state() == StateEnum.TRIAL

    def test_parent_time_continues_trial_without_reclassify(self):
        """『是』确认后共享状态=TRIAL,家长答时间 → 状态机直接续接 TrialAgent,不再分类。"""
        rec, trial, state_manager, classifier, processor = self._make_processor()

        # 第 1 轮:推荐 → 完成后状态置 RECOMMENDATION_CONFIRM(给下一轮用,本轮已走完分类)
        rec.input_parser.extract.return_value = _COMPLETE_INPUT
        rec._recommendation_service.recommend_teachers.return_value = _RECS
        rec._student_service.initialize_default_student.return_value = {"id": 1}
        _run(_collect(processor.process_task_stream("初二数学")))
        # 第 2 轮:选定老师 → 状态仍 RECOMMENDATION_CONFIRM,主路由跳过分类器直接回推荐 agent
        prior_calls = classifier.call_count
        _run(_collect(processor.process_task_stream("看高远不错")))
        assert classifier.call_count == prior_calls  # 关键:进入推荐确认态后,主路由不再分类
        # 第 3 轮:确认『是』→ 仍走推荐确认态路由(不分类),确认完成后共享状态置 TRIAL
        prior_calls = classifier.call_count
        _run(_collect(processor.process_task_stream("是")))
        assert classifier.call_count == prior_calls  # 本轮也不分类
        assert state_manager.get_current_state() == StateEnum.TRIAL

        # 第 4 轮:家长答时间 → 状态机直接续接 TrialAgent(不应再分类)
        prior_calls = classifier.call_count
        out = _run(_collect(processor.process_task_stream("周二晚上6点")))
        # 未重新分类
        assert classifier.call_count == prior_calls
        # 续接进入 TrialAgent(输出含试讲风格内容)
        assert trial.calls == ["周二晚上6点"]
        assert "试讲" in out
        # 首轮分流文案"这是一个试讲预约请求"只出现在 route_to_trial,续轮不带
        assert "这是一个试讲预约请求" not in out
        assert "这是一个老师推荐请求" not in out