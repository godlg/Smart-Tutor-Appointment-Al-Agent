"""LangGraph P0-A 主流程等价性测试。

覆盖 `.claude/tasks/langgraph-migrate-p0a.md` 的 9 个场景，与 legacy 行为逐条对齐：

- A1 `record_input` 被调用（含无关轮）
- A2 已转人工 → 短路且不分类
- A3 主动关键词 → 触发 handoff
- B1/B7 匿名校验（真实 `Identity(legacy=False)` + `set_current_identity`）
- B2/B3/B4 三个续轮态各自跳过分类器直连对应 agent（断言分类器 0 调用）
- B5 档期白名单 → 不调分类器、不改 flow_state
- B6 分类器抛 `ClassificationUnavailable` → BUSY 文案且踢回计数不增长
- D 踢回 3 次 → HANDOFF；1-2 次 → 能力列表
- C 四条主路由各落对节点

另断言固定文案与 legacy **逐字一致**（`_VERIFICATION_REPLY` / `_BUSY_REPLY` / HANDOFF）。

约束（与仓库其它测试一致）：无 pytest-asyncio，异步用例统一 `_run(_collect(...))`
同步包装；替身 agent / 替身 handoff_service，不触网、不连真实 DB。
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agents.graph import build_graph, new_state  # noqa: E402
from agents.graph.build import (  # noqa: E402
    route_after_classify,
    route_by_intent,
    route_entry,
    route_resume,
)
from agents.graph.nodes import (  # noqa: E402
    _BUSY_REPLY,
    _VERIFICATION_REPLY,
)
from agents.task_classification.agent_router import (  # noqa: E402
    AgentRouter,
    route_thought,
)
from agents.task_classification.classification_processor import (  # noqa: E402
    _BUSY_REPLY as LEGACY_BUSY_REPLY,
    _ERROR_BUDGET_REPLY,
    _ERROR_GENERIC_TEMPLATE,
    _VERIFICATION_REPLY as LEGACY_VERIFICATION_REPLY,
)
from agents.task_classification.state_manager import StateManager  # noqa: E402
from agents.task_classification.task_classifier import ClassificationUnavailable  # noqa: E402
from agents.context.types import ContextBudgetExceeded  # noqa: E402
from config.identity import (  # noqa: E402
    Identity,
    legacy_identity,
    reset_current_identity,
    set_current_identity,
)
from services.human_handoff_service import (  # noqa: E402
    HANDED_OFF_REPLY,
    HANDOFF_REPLY,
)


def _run(coro):
    return asyncio.run(coro)


async def _collect(graph, payload, cfg):
    out = []
    async for token in graph.astream(payload, cfg, stream_mode="custom"):
        out.append(token)
    return out


def _drain(graph, payload, cfg):
    return "".join(_run(_collect(graph, payload, cfg)))


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------

class FakeClassifier:
    """固定/按关键字映射返回意图，并记录调用（用于断言"未分类"）。"""

    def __init__(self, mapping=None, default="unrelated"):
        self.mapping = mapping or {}
        self.default = default
        self.calls = []

    async def classify_task(self, task):
        self.calls.append(task)
        for key, intent in self.mapping.items():
            if key in task:
                return intent
        return self.default


class RaisingClassifier:
    """固定抛 ClassificationUnavailable，并记录调用次数。"""

    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        raise ClassificationUnavailable("429")


class FakeSharedState:
    def __init__(self, value=None):
        self.value = value


class FakeAgent:
    """转发同形态替身：记录调用与 intent，按 token 产出文本，带可写共享状态。"""

    def __init__(self, name, tokens=None, state=None):
        self.name = name
        self.calls = []
        self.intents = []
        self.state = state
        self.tokens = tokens or [f"[REPLY][{name}]处理完成"]

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        self.intents.append(None)
        for token in self.tokens:
            yield token


class LessonLikeAgent(FakeAgent):
    """模拟真 LessonAgent：接收 intent 关键字（调课/退课/续轮共用）。"""

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        self.intents.append(intent)
        for token in self.tokens:
            yield token


class FakeAvailabilityHandler:
    def __init__(self, reply="[REPLY]张明老师最近可预约:周一 9:00"):
        self.reply = reply
        self.calls = []

    async def handle(self, task):
        self.calls.append(task)
        yield self.reply


class FakeHandoffService:
    """替身 handoff：进程内计数 + 记录触发，不落 DB。"""

    def __init__(self, handed=False, keyword=None, threshold=3):
        self.handed = handed
        self.keyword = keyword
        self.threshold = threshold
        self.records = []
        self.kickbacks = 0
        self.record_kickback_calls = 0
        self.triggered = []

    def record_input(self, session_id, content):
        self.records.append((session_id, content))

    def is_handed_off(self, session_id):
        return self.handed

    def parent_message(self, session_id, content):
        self.records.append(("parent", session_id, content))

    def match_keyword(self, task):
        return self.keyword if (self.keyword and task and self.keyword in task) else None

    def record_kickback(self, session_id):
        self.record_kickback_calls += 1
        self.kickbacks += 1
        return self.kickbacks >= self.threshold

    def get_count(self, session_id):
        return self.kickbacks

    def trigger_handoff(self, session_id, trigger_type, **kwargs):
        self.triggered.append((trigger_type, kwargs))

    @staticmethod
    def handoff_reply():
        return HANDOFF_REPLY

    @staticmethod
    def handed_off_reply():
        return HANDED_OFF_REPLY

    def build_transcript(self, session_id, messages):
        return None


ALL_BUSINESS = ("consultation", "recommendation", "trial", "lesson", "reschedule", "cancel")


def _agents(**overrides):
    agents = {key: FakeAgent(key) for key in ALL_BUSINESS}
    agents["lesson"] = LessonLikeAgent("lesson")
    agents["reschedule"] = LessonLikeAgent("reschedule")
    agents["cancel"] = LessonLikeAgent("cancel")
    agents.update(overrides)
    return agents


def _graph(classifier=None, agents=None, handoff=None, availability=None, router=None,
           history=None):
    return build_graph(
        classifier=classifier or FakeClassifier({}, default="unrelated"),
        agents=agents if agents is not None else _agents(),
        handoff_service=handoff,
        availability_handler=availability,
        agent_router=router,
        session_chat_history=history,
    )


def _with_identity(identity, func):
    token = set_current_identity(identity)
    try:
        return func()
    finally:
        reset_current_identity(token)


# ----------------------------------------------------------------------
# 文案常量：必须从原模块导入，与 legacy 逐字一致
# ----------------------------------------------------------------------

class TestCopyParity:

    def test_verification_reply_is_imported_not_copied(self):
        assert _VERIFICATION_REPLY == LEGACY_VERIFICATION_REPLY
        assert _VERIFICATION_REPLY.startswith("[REPLY]")

    def test_busy_reply_is_imported_not_copied(self):
        assert _BUSY_REPLY == LEGACY_BUSY_REPLY
        assert _BUSY_REPLY.startswith("[REPLY]")


# ----------------------------------------------------------------------
# A. 入口层
# ----------------------------------------------------------------------

class TestEntryLayer:

    def test_a1_record_input_is_called_including_unrelated_turn(self):
        hs = FakeHandoffService()
        graph = _graph(classifier=FakeClassifier({}, default="unrelated"), handoff=hs)
        _drain(graph, new_state("今天天气怎么样", "a1"), _cfg("a1"))
        # A1：无关轮也必须记录(供 transcript 兜底)
        assert hs.records == [("a1", "今天天气怎么样")]

    def test_a2_handed_off_short_circuits_without_classifying(self):
        hs = FakeHandoffService(handed=True)
        classifier = FakeClassifier({}, default="trial")
        graph = _graph(classifier=classifier, handoff=hs)
        out = _drain(graph, new_state("再问一句", "a2"), _cfg("a2"))
        assert HANDED_OFF_REPLY in out
        assert out.startswith("[REPLY]")
        assert "[HANDOFF]" not in out
        assert classifier.calls == []                     # 不进分类
        assert ("parent", "a2", "再问一句") in hs.records  # 家长消息落库

    def test_a3_keyword_triggers_handoff(self):
        hs = FakeHandoffService(keyword="人工")
        classifier = FakeClassifier({}, default="trial")
        graph = _graph(classifier=classifier, handoff=hs)
        out = _drain(graph, new_state("我要转人工", "a3"), _cfg("a3"))
        assert "[REPLY][HANDOFF]" in out
        assert HANDOFF_REPLY in out
        assert classifier.calls == []
        assert hs.triggered and hs.triggered[0][0] == "keyword"

    def test_a3_handoff_reply_is_verbatim(self):
        hs = FakeHandoffService(keyword="人工")
        graph = _graph(handoff=hs)
        out = _drain(graph, new_state("转人工", "a3b"), _cfg("a3b"))
        assert HANDOFF_REPLY in out


# ----------------------------------------------------------------------
# B. 分类前 / 分类后
# ----------------------------------------------------------------------

class TestAnonymousVerification:
    """B1/B7：匿名校验（用真实 Identity(legacy=False) + set_current_identity）。"""

    def test_b7_classified_business_request_requires_verification(self):
        classifier = FakeClassifier({}, default="trial")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)

        def _go():
            return _drain(graph, new_state("我要约课", "b7"), _cfg("b7"))

        out = _with_identity(Identity("anon_parent", student_id="1"), _go)
        assert _VERIFICATION_REPLY == LEGACY_VERIFICATION_REPLY
        assert _VERIFICATION_REPLY in out
        assert agents["trial"].calls == []                # 业务 agent 未被调用

    def test_b1_anonymous_business_state_requires_verification(self):
        classifier = FakeClassifier({}, default="consultation")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)

        def _go():
            # flow_state 已是业务态 → 入口即校验，不调分类器
            return _drain(graph, new_state("确认", "b1", flow_state="trial_confirm"),
                          _cfg("b1"))

        out = _with_identity(Identity("anon_parent", student_id="1"), _go)
        assert _VERIFICATION_REPLY in out
        assert classifier.calls == []

    def test_legacy_identity_still_routes_business(self):
        classifier = FakeClassifier({}, default="trial")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)

        def _go():
            return _drain(graph, new_state("我要约课", "b7-legacy"), _cfg("b7-legacy"))

        out = _with_identity(legacy_identity("b7-legacy"), _go)
        assert "[REPLY][trial]" in out
        assert agents["trial"].calls == ["我要约课"]


class TestContinuationStates:
    """B2/B3/B4：三个续轮态各自跳过分类器，直连对应 agent。"""

    @pytest.mark.parametrize("flow_state,target", [
        ("recommendation_confirm", "recommendation"),
        ("trial_confirm", "trial"),
        ("trial_convert", "trial"),
    ])
    def test_continuation_skips_classifier(self, flow_state, target):
        classifier = FakeClassifier({}, default="unrelated")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)
        tid = f"resume-{flow_state}"

        out = _drain(graph, new_state("确认", tid, flow_state=flow_state), _cfg(tid))

        assert classifier.calls == []                      # 分类器 0 调用
        assert agents[target].calls == ["确认"]
        assert "[REPLY]" in out

    @pytest.mark.parametrize("flow_state,target", [
        ("recommendation_confirm", "recommendation_node"),
        ("trial_confirm", "trial_node"),
        ("trial_convert", "trial_node"),
        ("trial", "trial_node"),
        ("lesson", "lesson_node"),
        ("consultation", "consultation_node"),
    ])
    def test_route_resume_pure_mapping(self, flow_state, target):
        assert route_resume({"flow_state": flow_state}) == target

    def test_route_resume_includes_trial_confirm_explicitly(self):
        """legacy 的 route_by_state 漏了 TRIAL_CONFIRM；图里必须显式表达。"""
        assert route_resume({"flow_state": "trial_confirm"}) == "trial_node"


class TestAvailabilityWhitelist:
    """B5：档期白名单直答，不调分类器、不改 flow_state。"""

    def test_availability_short_circuits_without_classifier_and_keeps_flow_state(self):
        classifier = FakeClassifier({}, default="consultation")
        availability = FakeAvailabilityHandler()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(),
                       availability=availability)
        tid = "b5"
        cfg = _cfg(tid)
        out = _drain(graph, new_state("张明老师什么时候有空", tid), cfg)

        assert availability.calls == ["张明老师什么时候有空"]
        assert classifier.calls == []
        assert "可预约" in out

        async def _state():
            return (await graph.aget_state(cfg)).values

        values = _run(_state())
        assert values["flow_state"] == "classify"          # 不改状态机


class TestClassificationUnavailableFallback:
    """B6：分类器抛 ClassificationUnavailable → BUSY 文案，踢回计数不增长。"""

    def test_busy_reply_and_no_kickback_growth(self):
        classifier = RaisingClassifier()
        hs = FakeHandoffService()
        graph = _graph(classifier=classifier, handoff=hs)
        out = _drain(graph, new_state("我想给孩子约一节数学试听课", "b6"), _cfg("b6"))

        assert _BUSY_REPLY in out
        assert out.startswith("[REPLY]")
        assert classifier.calls == 1
        assert hs.record_kickback_calls == 0               # 踢回计数不增长
        assert hs.kickbacks == 0
        assert hs.triggered == []                          # 不转人工

    def test_busy_keeps_flow_state_classify(self):
        graph = _graph(classifier=RaisingClassifier(), handoff=FakeHandoffService())
        tid = "b6-state"
        _drain(graph, new_state("随便一个问题", tid), _cfg(tid))

        async def _state():
            return (await graph.aget_state(_cfg(tid))).values

        assert _run(_state())["flow_state"] == "classify"


class TestUnrelatedFallback:
    """D：踢回计数达阈值 → HANDOFF；未达阈值 → 能力列表。"""

    def test_below_threshold_returns_capability_list(self):
        hs = FakeHandoffService(threshold=3)
        classifier = FakeClassifier({}, default="unrelated")
        graph = _graph(classifier=classifier, handoff=hs)
        tid = "d-low"

        first = _drain(graph, new_state("今天天气怎么样", tid), _cfg(tid))
        second = _drain(graph, new_state("随便聊聊", tid), _cfg(tid))

        assert "超出我的能力范围" in first
        assert "超出我的能力范围" in second
        assert "[HANDOFF]" not in first + second
        assert hs.triggered == []

    def test_third_kickback_triggers_handoff(self):
        hs = FakeHandoffService(threshold=3)
        classifier = FakeClassifier({}, default="unrelated")
        graph = _graph(classifier=classifier, handoff=hs)
        tid = "d-high"

        _drain(graph, new_state("天气一", tid), _cfg(tid))
        _drain(graph, new_state("天气二", tid), _cfg(tid))
        third = _drain(graph, new_state("天气三", tid), _cfg(tid))

        assert "[REPLY][HANDOFF]" in third
        assert HANDOFF_REPLY in third
        assert hs.triggered and hs.triggered[0][0] == "kickback_threshold"
        assert hs.triggered[0][1]["unrelated_count"] == 3


class TestBusinessRoutes:
    """C：主路由各落对节点（替身 agent 记录被调用）。"""

    @pytest.mark.parametrize("user_input,intent,node", [
        ("孩子数学怎么学", "consultation", "consultation"),
        ("想约一节试听课", "trial", "trial"),
        ("帮我预约明天下午3点的课", "lesson", "lesson"),
        ("把周三的课调到周五", "reschedule", "reschedule"),
        ("帮我取消明天的课", "cancel", "cancel"),
        ("请给我推荐一位老师", "recommendation", "recommendation"),
    ])
    def test_route_reaches_expected_agent(self, user_input, intent, node):
        classifier = FakeClassifier({}, default=intent)
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)
        tid = f"c-{node}"

        _drain(graph, new_state(user_input, tid), _cfg(tid))

        assert agents[node].calls == [user_input]
        for other in ALL_BUSINESS:
            if other != node:
                assert agents[other].calls == []

    def test_reschedule_and_cancel_pass_intent(self):
        classifier = FakeClassifier({}, default="reschedule")
        agents = _agents()
        graph = _graph(classifier=classifier, handoff=FakeHandoffService(), agents=agents)

        _drain(graph, new_state("把周三的课调到周五", "c-intent-r"), _cfg("c-intent-r"))
        assert agents["reschedule"].intents == ["reschedule"]

        classifier2 = FakeClassifier({}, default="cancel")
        agents2 = _agents()
        graph2 = _graph(classifier=classifier2, handoff=FakeHandoffService(), agents=agents2)
        _drain(graph2, new_state("帮我取消明天的课", "c-intent-c"), _cfg("c-intent-c"))
        assert agents2["cancel"].intents == ["cancel"]

    def test_consultation_and_recommendation_flow_state(self):
        """consultation 完成后回 CLASSIFY；trial/lesson 置对应业务态（与 legacy 一致）。"""
        agents = _agents()
        graph = _graph(classifier=FakeClassifier({}, default="trial"), handoff=FakeHandoffService(),
                       agents=agents)
        tid = "c-flow-trial"
        _drain(graph, new_state("想约试听课", tid), _cfg(tid))

        async def _state():
            return (await graph.aget_state(_cfg(tid))).values

        assert _run(_state())["flow_state"] == "trial"


# ----------------------------------------------------------------------
# 纯函数路由
# ----------------------------------------------------------------------

class TestRoutePureFunctions:

    @pytest.mark.parametrize("state,expected", [
        ({"handoff": True}, "handoff_node"),
        ({"verification_required": True}, "verification_node"),
        ({"flow_state": "trial_confirm"}, "resume_node"),
        ({"flow_state": "classify", "messages": []}, "classify_node"),
    ])
    def test_route_entry(self, state, expected):
        assert route_entry(state) == expected

    def test_route_entry_availability_before_classify(self):
        state = {"flow_state": "classify",
                 "messages": [{"role": "human", "content": "张明老师什么时候有空"}]}
        assert route_entry(state) == "availability_node"

    @pytest.mark.parametrize("state,expected", [
        ({"busy": True}, "busy_node"),
        ({"verification_required": True, "intent": "trial"}, "verification_node"),
        ({"intent": "trial"}, "trial_node"),
        ({"intent": "lesson"}, "lesson_node"),
        ({"intent": "cancel"}, "cancel_node"),
        ({"intent": "package"}, "consultation_node"),
        ({"intent": "progress"}, "consultation_node"),
        ({"intent": "recommendation_confirm"}, "recommendation_node"),
        ({"intent": ""}, "unrelated_node"),
        ({"intent": "unrelated"}, "unrelated_node"),
    ])
    def test_route_after_classify(self, state, expected):
        assert route_after_classify(state) == expected

    def test_route_by_intent_poc_semantics_preserved(self):
        """既有 PoC 对外语义不变：只映射 4 条业务路由，其余兜底。"""
        assert route_by_intent({"intent": "cancel"}) == "unrelated_node"
        assert route_by_intent({"intent": "trial"}) == "trial_node"


# ----------------------------------------------------------------------
# D1：业务节点首轮分流行——复用 AgentRouter 文案，与 legacy 逐字一致
# ----------------------------------------------------------------------

class FakeConsultStreamer:
    """legacy AgentRouter.route_to_consultation 所需的异步上下文 + consult_stream 替身。"""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def consult_stream(self, task):
        self.calls.append(task)
        yield "[REPLY][consultation]处理完成"


def _first_token(gen):
    async def _inner():
        async for token in gen:
            return token
        return None

    return _run(_inner())


class TestD1RouteThoughtParity:
    """route_thought 与 legacy AgentRouter.route_to_* 的首行逐字一致（单一真源）。"""

    @pytest.mark.parametrize("method,intent", [
        ("route_to_trial", "trial"),
        ("route_to_lesson", "lesson"),
        ("route_to_reschedule", "reschedule"),
        ("route_to_cancel", "cancel"),
        ("route_to_recommendation", "recommendation"),
        ("route_to_consultation", "consultation"),
    ])
    def test_route_thought_matches_legacy_router_first_line(self, method, intent):
        router = AgentRouter(
            consultant_agent=FakeConsultStreamer(),
            trial_agent=FakeAgent("trial"),
            lesson_agent=LessonLikeAgent("lesson"),
            recommendation_agent=FakeAgent("recommendation"),
            state_manager=StateManager(),
        )
        first = _first_token(getattr(router, method)("任意输入"))
        assert first == route_thought(intent)

    def test_reschedule_and_cancel_use_distinct_copy(self):
        assert route_thought("reschedule") != route_thought("cancel")
        assert "调课" in route_thought("reschedule")
        assert "退课" in route_thought("cancel")
        assert route_thought("reschedule").startswith("[THOUGHT][归类机器人] ")

    @pytest.mark.parametrize("intent,node", [
        ("consultation", "consultation"),
        ("recommendation", "recommendation"),
        ("trial", "trial"),
        ("lesson", "lesson"),
        ("reschedule", "reschedule"),
        ("cancel", "cancel"),
    ])
    def test_first_turn_business_node_emits_legacy_route_thought(self, intent, node):
        agents = _agents()
        graph = _graph(classifier=FakeClassifier({}, default=intent),
                       handoff=FakeHandoffService(), agents=agents)
        tid = f"d1-{intent}"
        out = _drain(graph, new_state("随便一句话", tid), _cfg(tid))

        expected = (f"[THOUGHT][LangGraph] 意图识别为 {intent}。"
                    + route_thought(intent)
                    + "".join(agents[node].tokens))
        assert out == expected

    def test_continuation_state_has_no_route_thought(self):
        """续轮态直连（TRIAL_CONFIRM）不应再出现『又来一个试讲预约请求』分流行。"""
        agents = _agents()
        graph = _graph(classifier=FakeClassifier({}, default="unrelated"),
                       handoff=FakeHandoffService(), agents=agents)
        tid = "d1-resume"
        out = _drain(graph, new_state("确认", tid, flow_state="trial_confirm"), _cfg(tid))

        assert route_thought("trial") not in out
        assert out == "".join(agents["trial"].tokens)


# ----------------------------------------------------------------------
# D3：咨询节点经 async with 进入，触发知识库初始化
# ----------------------------------------------------------------------

class FakeConsultant:
    """模拟 ConsultantAgent：__aenter__ 触发知识库初始化（可观测）。"""

    def __init__(self, tokens=None):
        self.entered = 0
        self.exited = 0
        self.kb_initialized = False
        self.calls = []
        self.tokens = tokens or ["[REPLY][consultation]处理完成"]

    async def __aenter__(self):
        self.entered += 1
        self.kb_initialized = True          # 对应 knowledge_retriever.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        for token in self.tokens:
            yield token


class TestD3ConsultationInitialization:

    def test_consultation_node_enters_agent_via_aenter(self):
        consultant = FakeConsultant()
        agents = {**_agents(), "consultation": consultant}
        graph = _graph(classifier=FakeClassifier({}, default="consultation"),
                       handoff=FakeHandoffService(), agents=agents)
        tid = "d3-enter"
        out = _drain(graph, new_state("孩子数学怎么学", tid), _cfg(tid))

        assert consultant.entered == 1
        assert consultant.exited == 1
        assert consultant.kb_initialized is True    # 知识库被初始化
        assert consultant.calls == ["孩子数学怎么学"]
        assert route_thought("consultation") in out

    def test_consultation_continuation_also_enters_agent(self):
        """CONSULTATION 续轮（route_by_state 语义）也必须走 async with 初始化。"""
        consultant = FakeConsultant()
        agents = {**_agents(), "consultation": consultant}
        graph = _graph(classifier=FakeClassifier({}, default="unrelated"),
                       handoff=FakeHandoffService(), agents=agents)
        tid = "d3-resume"
        _drain(graph, new_state("继续刚才的问题", tid, flow_state="consultation"), _cfg(tid))

        assert consultant.entered == 1
        assert consultant.kb_initialized is True
        assert consultant.calls == ["继续刚才的问题"]

    def test_lightweight_stub_without_aenter_still_streams(self):
        """无 __aenter__ 的轻量替身不应因 D3 改动而崩（保持可注入性）。"""
        stub = FakeAgent("consultation")
        agents = {**_agents(), "consultation": stub}
        graph = _graph(classifier=FakeClassifier({}, default="consultation"),
                       handoff=FakeHandoffService(), agents=agents)
        tid = "d3-stub"
        out = _drain(graph, new_state("数学怎么学", tid), _cfg(tid))

        assert stub.calls == ["数学怎么学"]
        assert "[REPLY][consultation]" in out

    def test_adapter_forwards_aenter_to_consultant(self):
        """生产适配器 _ConsultStreamAdapter 透传 __aenter__ 触发知识库初始化。"""
        from api.chat_handler import _ConsultStreamAdapter

        consultant = FakeConsultant()
        adapter = _ConsultStreamAdapter(consultant)

        async def _go():
            async with adapter as entered:
                assert entered is adapter

        _run(_go())
        assert consultant.entered == 1
        assert consultant.kb_initialized is True


# ----------------------------------------------------------------------
# D4：图路径异常兜底（等价 legacy process_task_stream:169-173）
# ----------------------------------------------------------------------

class BudgetClassifier:
    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        raise ContextBudgetExceeded("超预算", 100, 10)


class BoomClassifier:
    def __init__(self, exc=None):
        self.calls = 0
        self.exc = exc or RuntimeError("boom")

    async def classify_task(self, task):
        self.calls += 1
        raise self.exc


class ExplodingAgent(FakeAgent):
    def __init__(self, name, exc):
        super().__init__(name)
        self.exc = exc

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        raise self.exc
        yield  # pragma: no cover - 使本方法成为异步生成器


class TestD4ErrorFallback:

    def test_context_budget_exceeded_emits_legacy_reply(self):
        classifier = BudgetClassifier()
        hs = FakeHandoffService()
        graph = _graph(classifier=classifier, handoff=hs)
        tid = "d4-budget"
        out = _drain(graph, new_state("很长很长的问题", tid), _cfg(tid))

        assert out == _ERROR_BUDGET_REPLY
        assert _ERROR_BUDGET_REPLY == "[ERROR]上下文超过安全预算，请缩短输入后重试。"
        assert "超出我的能力范围" not in out         # 不落到 unrelated 兜底
        assert hs.record_kickback_calls == 0         # 不踢回

    def test_generic_exception_emits_legacy_template_and_resets_flow_state(self):
        graph = _graph(classifier=BoomClassifier(RuntimeError("boom")),
                       handoff=FakeHandoffService())
        tid = "d4-boom"
        out = _drain(graph, new_state("随便一句话", tid), _cfg(tid))

        assert out == _ERROR_GENERIC_TEMPLATE.format(error=RuntimeError("boom"))
        assert out == "[ERROR]处理任务时发生错误: boom"

        async def _state():
            return (await graph.aget_state(_cfg(tid))).values

        assert _run(_state())["flow_state"] == "classify"   # force_reset 语义

    def test_business_agent_exception_is_wrapped(self):
        agents = {**_agents(), "trial": ExplodingAgent("trial", RuntimeError("agent-down"))}
        graph = _graph(classifier=FakeClassifier({}, default="trial"),
                       handoff=FakeHandoffService(), agents=agents)
        tid = "d4-agent"
        out = _drain(graph, new_state("帮我约一节试听课", tid), _cfg(tid))

        assert route_thought("trial") in out
        assert out.endswith("[ERROR]处理任务时发生错误: agent-down")

        async def _state():
            return (await graph.aget_state(_cfg(tid))).values

        assert _run(_state())["flow_state"] == "classify"

    def test_error_short_circuits_downstream_node(self):
        hs = FakeHandoffService()
        graph = _graph(classifier=BoomClassifier(), handoff=hs)
        tid = "d4-short"
        out = _drain(graph, new_state("随便一句话", tid), _cfg(tid))

        assert "[ERROR]" in out
        assert "超出我的能力范围" not in out
        assert hs.record_kickback_calls == 0

    def test_next_turn_recovers_after_error(self):
        class FlakyClassifier:
            def __init__(self):
                self.calls = 0

            async def classify_task(self, task):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("第一次失败")
                return "trial"

        agents = _agents()
        graph = _graph(classifier=FlakyClassifier(), handoff=FakeHandoffService(),
                       agents=agents)
        tid = "d4-recover"
        first = _drain(graph, new_state("第一次", tid), _cfg(tid))
        assert "[ERROR]" in first

        second = _drain(graph, new_state("第二次", tid), _cfg(tid))
        assert "[REPLY][trial]" in second
        assert agents["trial"].calls == ["第二次"]

    @pytest.mark.parametrize("state,expected", [
        ({"error": True, "intent": "trial"}, "error_node"),
        ({"error": False, "intent": "trial"}, "trial_node"),
    ])
    def test_route_after_classify_error_flag(self, state, expected):
        assert route_after_classify(state) == expected

    def test_error_flag_routes_pure_functions(self):
        assert route_entry({"error": True, "handoff": True}) == "error_node"
        assert route_resume({"error": True, "flow_state": "trial"}) == "error_node"
