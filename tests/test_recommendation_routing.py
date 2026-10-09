"""
老师推荐路由测试（修复 chat 入口"推荐老师"路径）

注：本仓库未装 pytest-asyncio,异步用例统一用同步包装 _run(coro)
(参照 tests/test_reschedule_routing.py 的 _run 约定)。
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from agents.recommendation_agent import (
    RecommendationAgent,
    RecommendationInputParser,
)
from agents.task_classification.agent_router import AgentRouter


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream):
    tokens = []
    async for tok in stream:
        tokens.append(tok)
    return "".join(tokens)


# ----------------------------------------------------------------------
# 1. RecommendationInputParser:JSON 解析容错
# ----------------------------------------------------------------------

class TestRecommendationParser:

    def test_extract_parses_valid_json(self):
        parser = RecommendationInputParser.__new__(RecommendationInputParser)
        parser.chain = MagicMock()
        parser.chain.invoke.return_value = MagicMock(
            content='{"subject": "数学", "grade_level": "初中", "info_complete": true}'
        )
        out = parser.extract("孩子初二,数学")
        assert out["subject"] == "数学"
        assert out["grade_level"] == "初中"
        assert out["info_complete"] is True

    def test_extract_fallback_on_json_error(self):
        parser = RecommendationInputParser.__new__(RecommendationInputParser)
        parser.chain = MagicMock()
        parser.chain.invoke.side_effect = RuntimeError("boom")
        out = parser.extract("任意输入")
        assert out["subject"] == "未知"
        assert out["info_complete"] is False


# ----------------------------------------------------------------------
# 2. RecommendationAgent.run_stream:字段不全追问 / 全则打分
# ----------------------------------------------------------------------

class TestRecommendationAgentRun:

    def _make_agent(self, parser_output):
        agent = RecommendationAgent.__new__(RecommendationAgent)
        agent.session_id = "s1"
        agent.message_builder = RecommendationAgent(
            session_id="builder-source"
        ).message_builder
        agent.input_parser = MagicMock()
        agent.input_parser.extract.return_value = parser_output
        agent._recommendation_service = MagicMock()
        agent._student_service = MagicMock()
        return agent

    def test_missing_fields_asks_clarifying_question(self):
        agent = self._make_agent({
            "subject": "未知", "grade_level": "未知", "preferred_style": "未知",
            "preferred_teacher_name": "未知", "mode": "未知",
            "info_complete": False, "missing_info": ["subject", "grade_level"],
        })
        out = _run(_collect(agent.run_stream("推荐个老师")))
        assert "[推荐助手]" in out
        assert "学科" in out
        agent._recommendation_service.recommend_teachers.assert_not_called()

    def test_complete_fields_calls_recommend_teachers_and_lists(self):
        agent = self._make_agent({
            "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
            "preferred_teacher_name": "未知", "mode": "未知",
            "info_complete": True, "missing_info": [],
        })
        agent._recommendation_service.recommend_teachers.return_value = [
            {"teacher": {"id": 7, "name": "张老师"}, "score": 90, "reason": "擅长初中数学"},
            {"teacher": {"id": 9, "name": "李老师"}, "score": 75, "reason": "覆盖初二"},
        ]
        agent._student_service.initialize_default_student.return_value = {"id": 1}

        out = _run(_collect(agent.run_stream("初二,数学")))
        agent._recommendation_service.recommend_teachers.assert_called_once()
        assert "张老师" in out
        assert "ID 7" in out

    def test_service_failure_yields_graceful_message(self):
        agent = self._make_agent({
            "subject": "数学", "grade_level": "初中", "preferred_style": "未知",
            "preferred_teacher_name": "未知", "mode": "未知",
            "info_complete": True, "missing_info": [],
        })
        agent._recommendation_service.recommend_teachers.side_effect = RuntimeError("db down")

        out = _run(_collect(agent.run_stream("初二数学")))
        assert "[推荐助手]" in out
        assert "推荐服务暂时不可用" in out


# ----------------------------------------------------------------------
# 3. AgentRouter.route_to_recommendation
# ----------------------------------------------------------------------

class _FakeRecommendationAgent:
    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][推荐助手]推荐结果"


class TestAgentRouterRecommendation:

    def test_route_to_recommendation_passes_user_input(self):
        agent = _FakeRecommendationAgent()
        router = AgentRouter(recommendation_agent=agent, state_manager=MagicMock())
        out = _run(_collect(router.route_to_recommendation("推荐个老师")))
        assert agent.calls == ["推荐个老师"]
        assert "推荐结果" in out

    def test_route_to_recommendation_without_agent_yields_error(self):
        router = AgentRouter(recommendation_agent=None, state_manager=MagicMock())
        out = _run(_collect(router.route_to_recommendation("推荐")))
        assert "[ERROR]" in out


# ----------------------------------------------------------------------
# 4. RecommendationAgent 确认状态机(修"高远不错"被误归到追问)
# ----------------------------------------------------------------------

# 固定的一次推荐:字段齐全 → 返回 Top-3,并把确认态推进到 awaiting_confirmation=True
_RECS = [
    {"teacher": {"id": 7, "name": "高远"}, "score": 92, "reason": "擅长初二数学"},
    {"teacher": {"id": 9, "name": "张明"}, "score": 80, "reason": "覆盖初中"},
    {"teacher": {"id": 11, "name": "王芳"}, "score": 70, "reason": "耐心细致"},
]
_COMPLETE_INPUT = {
    "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
    "preferred_teacher_name": "未知", "mode": "未知",
    "info_complete": True, "missing_info": [],
}


def _make_confirm_agent():
    """构造一个已完成一轮推荐的 agent(进入确认态),并紧贴该会话的 history。"""
    agent = RecommendationAgent.__new__(RecommendationAgent)
    agent.session_id = "s-confirm"
    agent.chats_by_session_id = {}
    agent.message_builder = RecommendationAgent(session_id="b").message_builder
    agent.input_parser = MagicMock()
    agent._recommendation_service = MagicMock()
    agent._student_service = MagicMock()
    # __new__ 构造缺 self.state,显式给一个共享状态(置 TRIAL 时断言用)
    agent.state = SharedState()
    return agent


def _recommend(agent, sid="s-confirm"):
    """跑一轮完整推荐,返回该会话的 history(dict)。"""
    agent.input_parser.extract.return_value = _COMPLETE_INPUT
    agent._recommendation_service.recommend_teachers.return_value = _RECS
    agent._student_service.initialize_default_student.return_value = {"id": 1}
    _run(_collect(agent.run_stream("初二数学", session_id=sid)))
    return agent.history[sid]


class TestRecommendationConfirmation:

    def test_complete_fields_stores_history_and_awaits(self):
        """字段齐全 → Top-N + 存 last_recommendations + awaiting=True"""
        agent = _make_confirm_agent()
        hist = _recommend(agent)
        assert hist["awaiting_confirmation"] is True
        assert hist["confirmed_teacher"] is None
        assert len(hist["last_recommendations"]) == 3
        assert hist["last_recommendations"][0]["teacher"]["name"] == "高远"

    def test_missing_fields_keeps_awaiting_false(self):
        """追问后 history.awaiting_confirmation 仍为 False(避免误进确认分支)"""
        agent = _make_confirm_agent()
        agent.input_parser.extract.return_value = {
            "subject": "未知", "grade_level": "未知", "preferred_style": "未知",
            "preferred_teacher_name": "未知", "mode": "未知",
            "info_complete": False, "missing_info": ["subject", "grade_level"],
        }
        out = _run(_collect(agent.run_stream("推荐个老师", session_id="s-confirm")))
        assert "学科" in out
        assert agent.history["s-confirm"]["awaiting_confirmation"] is False

    def test_confirm_by_name_matches_and_sets_confirmed_teacher(self):
        """确认态输入'高远' → 名字匹配 + yield 选定消息 + confirmed_teacher 设置"""
        agent = _make_confirm_agent()
        _recommend(agent)
        out = _run(_collect(agent.run_stream("看高远不错", session_id="s-confirm")))
        hist = agent.history["s-confirm"]
        assert "已为您选定 高远 老师" in out
        assert hist["confirmed_teacher"]["teacher"]["name"] == "高远"
        assert "预约试讲请回复" in out

    def test_confirm_by_index_matches(self):
        """确认态输入'1' → 按索引匹配第一顺位"""
        agent = _make_confirm_agent()
        _recommend(agent)
        out = _run(_collect(agent.run_stream("1", session_id="s-confirm")))
        assert agent.history["s-confirm"]["confirmed_teacher"]["teacher"]["name"] == "高远"

    def test_confirm_by_teacher_id_matches(self):
        """确认态输入'ID 7' → 按老师 id 匹配"""
        agent = _make_confirm_agent()
        _recommend(agent)
        out = _run(_collect(agent.run_stream("就选 ID 7 吧", session_id="s-confirm")))
        assert agent.history["s-confirm"]["confirmed_teacher"]["teacher"]["id"] == 7

    def test_confirm_with_unknown_teacher_asks_who(self):
        """确认态输入'张三'(不在候选) → 引导'您说的是哪位?'"""
        agent = _make_confirm_agent()
        _recommend(agent)
        out = _run(_collect(agent.run_stream("张三", session_id="s-confirm")))
        assert "您说的是哪位" in out
        assert agent.history["s-confirm"]["confirmed_teacher"] is None

    def test_confirm_yes_leads_directly_to_trial(self):
        """确认态输入'是'(已选定老师) → 直接预约试讲问时间,不再问『试讲/正式课』"""
        from config.constants import StateEnum
        agent = _make_confirm_agent()
        hist = _recommend(agent)
        # 先选定老师
        _run(_collect(agent.run_stream("高远", session_id="s-confirm")))
        assert hist["confirmed_teacher"] is not None
        out = _run(_collect(agent.run_stream("是", session_id="s-confirm")))
        # 直接引导试讲时间,不再重复问"回复『试讲』"
        assert "什么时候试讲" in out
        assert "回复『试讲』" not in out
        assert "『正式课』" not in out
        # 上下文载体(学科/年级/老师ID)保留,下游 TrialAgent 可续接
        assert "学科: 数学" in out
        assert "年级: 初中" in out
        assert "老师ID:" in out
        # 共享状态置 TRIAL,下一轮由状态机直接续接 TrialAgent
        assert agent.state.value == StateEnum.TRIAL
        # 确认完成后清空确认态,保证新推荐不会误进
        assert hist["confirmed_teacher"] is None
        assert hist["awaiting_confirmation"] is False

    def test_confirm_negative_clears_and_guides_reselect(self):
        """确认态输入'不' → 清掉 confirmed_teacher + awaiting,进入重选引导"""
        agent = _make_confirm_agent()
        hist = _recommend(agent)
        _run(_collect(agent.run_stream("高远", session_id="s-confirm")))
        out = _run(_collect(agent.run_stream("不", session_id="s-confirm")))
        assert hist["confirmed_teacher"] is None
        assert hist["awaiting_confirmation"] is False
        assert hist["recommendation_declined"] is True
        assert "重新推荐" in out

    def test_positive_without_selection_confirms_first(self):
        """确认态尚未选老师,直接回'好' → 默认选第一顺位"""
        agent = _make_confirm_agent()
        _recommend(agent)
        out = _run(_collect(agent.run_stream("好", session_id="s-confirm")))
        assert agent.history["s-confirm"]["confirmed_teacher"]["teacher"]["name"] == "高远"


# ----------------------------------------------------------------------
# 5. 主控意图分类: recommendation_confirm
# ----------------------------------------------------------------------

from agents.task_classification.task_classifier import TaskClassifier, _VALID_CATEGORIES


class TestRecommendationConfirmClassification:

    def test_valid_categories_include_confirm(self):
        assert "recommendation_confirm" in _VALID_CATEGORIES

    def test_extract_normalizes_confirm(self):
        assert TaskClassifier._extract_category("recommendation_confirm") == "recommendation_confirm"
        assert TaskClassifier._extract_category("recommendation_confirm。") == "recommendation_confirm"
        assert TaskClassifier._extract_category("9. recommendation_confirm") == "recommendation_confirm"

    def test_recommendation_confirm_differs_from_recommendation(self):
        """'看高远不错' 归推荐确认;'请给我推荐一位老师' 仍是推荐。"""
        assert TaskClassifier._extract_category("recommendation_confirm") == "recommendation_confirm"
        assert TaskClassifier._extract_category("recommendation") == "recommendation"


# ----------------------------------------------------------------------
# 6. ClassificationProcessor: recommendation_confirm 复用推荐 agent
# ----------------------------------------------------------------------

from agents.task_classification.classification_processor import ClassificationProcessor
from agents.task_classification.state_manager import (
    SharedState,
    StateManager,
)
from agents.task_classification.unrelated_handler import UnrelatedHandler


class _AsyncFakeClassifier:
    """固定返回某类别的异步分类器。"""

    def __init__(self, category):
        self.category = category

    async def classify_task(self, task):
        return self.category


class _ConfirmRouter:
    """带 recommendation_agent 的假路由器,记录是否路由到推荐 agent。"""

    def __init__(self):
        self.recommendation_agent = object()
        self.consultant_agent = None
        self.trial_agent = None
        self.lesson_agent = None
        self.called = False
        self.confirm_called = False

    async def route_to_recommendation(self, task):
        self.called = True
        yield "[REPLY][推荐助手]推荐处理结果"

    async def route_to_recommendation_confirm(self, task):
        """推荐确认态续轮(新增)"""
        self.confirm_called = True
        yield "[REPLY][推荐助手]推荐确认续轮结果"


class TestRecommendationConfirmRouting:

    def _make_processor(self, category="recommendation_confirm"):
        classifier = _AsyncFakeClassifier(category)
        state_manager = MagicMock()
        state_manager.should_classify.return_value = True
        # 默认不在『推荐确认态』:走原『分类→路由』路径(老测试要验证的语义)
        state_manager.is_in_recommendation_confirm_flow.return_value = False
        router = _ConfirmRouter()
        processor = ClassificationProcessor(
            task_classifier=classifier,
            state_manager=state_manager,
            agent_router=router,
            unrelated_handler=MagicMock(spec=UnrelatedHandler),
        )
        return processor, router

    def test_confirm_routes_to_recommendation_agent(self):
        """'高远不错' 分类为 recommendation_confirm → 走推荐 agent"""
        processor, router = self._make_processor("recommendation_confirm")
        out = _run(_collect(processor.process_task_stream("看高远不错")))
        assert router.called is True
        assert "推荐处理结果" in out

    def test_confirm_sync_collects_full_stream(self):
        """同步版 process_task_sync 也应路由并返回完整串"""
        processor, router = self._make_processor("recommendation_confirm")
        out = _run(processor.process_task_sync("看高远不错"))
        assert router.called is True
        assert "推荐处理结果" in out

    def test_recommendation_still_routes(self):
        """原 recommendation 意图不受影响"""
        processor, router = self._make_processor("recommendation")
        out = _run(_collect(processor.process_task_stream("请推荐一位老师")))
        assert router.called is True
        assert "推荐处理结果" in out
