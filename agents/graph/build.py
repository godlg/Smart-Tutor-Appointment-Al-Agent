"""LangGraph 主控编排：图构建（P0-A：图承载完整主流程）。

结构（线性前置 + 决策点，与 legacy 主流程 1:1 对应）：

    START -> entry_node                     # A1/A2/A3 + B1 匿名校验判定
    entry_node --route_entry--> { handoff_node | verification_node |
                                  availability_node | resume_node | classify_node }
    resume_node --route_resume--> { recommendation_node | consultation_node |
                                    trial_node | lesson_node | unrelated_node }
    classify_node --route_after_classify--> { busy_node | verification_node |
                                              consultation_node | recommendation_node |
                                              trial_node | lesson_node | reschedule_node |
                                              cancel_node | unrelated_node }
    各业务节点 / handoff_node / verification_node / busy_node / unrelated_node -> END

路由函数全部是**纯函数**（只读 state，或读 build 时固定的映射；不查 DB、不调 LLM），
便于参数化单测。

依赖注入：classifier / agents / handoff_service / availability_handler / agent_router
都由调用方传入；单测与离线 demo 用轻量替身即可跑通，无需初始化知识库 / LLM / DB。
checkpointer 默认 InMemorySaver。
"""

import logging
from functools import partial
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes as _nodes
from .state import TutorState

logger = logging.getLogger(__name__)


# flow_state 字面量：与 agents.task_classification.state_manager.FlowState 的 value 对齐。
# 注意 config.constants.StateEnum **不含** trial_confirm / trial_convert，
# 故这里用字符串常量而非枚举成员，避免误引用不存在的属性。
CLASSIFY = "classify"
CONSULTATION = "consultation"
RECOMMENDATION_CONFIRM = "recommendation_confirm"
TRIAL = "trial"
TRIAL_CONFIRM = "trial_confirm"
TRIAL_CONVERT = "trial_convert"
LESSON = "lesson"

# intent -> 业务节点名（纯映射，不查 DB / 不调 LLM）
# 覆盖 legacy `ClassificationProcessor` 第 126-148 行的全部 10 类意图。图的路由真源。
_INTENT_ROUTES = {
    "consultation": "consultation_node",
    "package": "consultation_node",
    "progress": "consultation_node",
    "recommendation": "recommendation_node",
    "recommendation_confirm": "recommendation_node",
    "trial": "trial_node",
    "lesson": "lesson_node",
    "reschedule": "reschedule_node",
    "cancel": "cancel_node",
    "unrelated": "unrelated_node",
}

# 保留 PoC 时代 `route_by_intent` 的对外语义（只映射 4 条业务路由，其余兜底）。
# 既有 PoC 测试依赖该语义；图内部统一走 `route_after_classify`。
_POC_INTENT_ROUTES = {
    "consultation": "consultation_node",
    "trial": "trial_node",
    "lesson": "lesson_node",
    "reschedule": "reschedule_node",
}

# flow_state -> 续轮目标节点（对应 legacy B2/B3/B4 + route_by_state）。
# 显式覆盖 TRIAL_CONFIRM / TRIAL_CONVERT / RECOMMENDATION_CONFIRM：
# legacy 的 route_by_state **漏了 TRIAL_CONFIRM**（靠更早的 B3 早返回兜住），
# 图里必须显式表达，不留隐式依赖。
_RESUME_ROUTES = {
    RECOMMENDATION_CONFIRM: "recommendation_node",
    CONSULTATION: "consultation_node",
    TRIAL: "trial_node",
    TRIAL_CONFIRM: "trial_node",
    TRIAL_CONVERT: "trial_node",
    LESSON: "lesson_node",
}

_FALLBACK_NODE = "unrelated_node"

_BUSINESS_KEYS = ("consultation", "recommendation", "trial", "lesson", "reschedule", "cancel")
_BUSINESS_NODES = tuple(f"{key}_node" for key in _BUSINESS_KEYS)


class TaskClassifierAdapter:
    """默认分类器：把惰性构造的 TaskClassifier.classify_task 包成可注入依赖。

    只有真正调用时才会构造底层 TaskClassifier（进而构造 LLM），因此
    注入替身 / 仅测路由纯函数时不会触发任何外部初始化。
    """

    def __init__(self, klass=None, llm=None):
        self._klass = klass
        self._llm = llm
        self._classifier = None

    def _ensure(self):
        if self._classifier is None:
            if self._klass is None:
                from agents.task_classification.task_classifier import TaskClassifier

                self._klass = TaskClassifier
            if self._llm is None:
                from config.model_provider import create_chat_model

                self._llm = create_chat_model(temperature=0)
            self._classifier = self._klass(self._llm)
        return self._classifier

    async def classify_task(self, task: str) -> str:
        return await self._ensure().classify_task(task)


# ----------------------------------------------------------------------
# 纯函数路由
# ----------------------------------------------------------------------

def route_by_intent(state: TutorState) -> str:
    """分类结果 -> 目标节点名（PoC 对外语义：4 条业务路由，其余兜底）。

    保留以兼容既有 PoC 测试；图内部统一走 `route_after_classify`（覆盖 10 类）。
    """
    return _POC_INTENT_ROUTES.get(state.get("intent") or "", _FALLBACK_NODE)


def route_entry(state: TutorState) -> str:
    """入口分支（纯函数）：异常兜底 > 转人工短路 > 匿名校验 > 续轮态 > 档期白名单 > 分类。

    顺序与 legacy 对齐：
    - D4 异常兜底（上游节点 error=True → 终止本轮）
    - A2/A3 转人工短路（entry_node 已置 handoff）
    - B1 匿名 + 业务态
    - B2/B3/B4 续轮态（flow_state != classify 直连，跳过分类器）
    - B5 档期白名单
    - 否则分类
    """
    if state.get("error"):
        return "error_node"
    if state.get("handoff"):
        return "handoff_node"
    if state.get("verification_required"):
        return "verification_node"
    flow_state = state.get("flow_state")
    if flow_state and flow_state != CLASSIFY:
        return "resume_node"
    from ..availability.availability_handler import is_availability_query

    if is_availability_query(_nodes.last_human_text(state)):
        return "availability_node"
    return "classify_node"


def route_resume(state: TutorState, resume_routes: Optional[dict] = None) -> str:
    """续轮态分支（纯函数）：异常兜底 > flow_state 显式映射到对应业务节点。"""
    if state.get("error"):
        return "error_node"
    routes = _RESUME_ROUTES if resume_routes is None else resume_routes
    return routes.get(state.get("flow_state") or "", _FALLBACK_NODE)


def route_after_classify(state: TutorState, intent_routes: Optional[dict] = None) -> str:
    """分类后分支（纯函数）：异常兜底 > busy > 匿名校验 > intent 映射 > 兜底。"""
    if state.get("error"):
        return "error_node"
    if state.get("busy"):
        return "busy_node"
    if state.get("verification_required"):
        return "verification_node"
    routes = _INTENT_ROUTES if intent_routes is None else intent_routes
    return routes.get(state.get("intent") or "", _FALLBACK_NODE)


def route_after_unrelated(state: TutorState) -> str:
    """D 兜底后分支（纯函数）：踢回达阈值已置 handoff=True → 挂起节点；否则 END。"""
    if state.get("handoff"):
        return "handoff_wait_node"
    return END


# ----------------------------------------------------------------------
# 图构建
# ----------------------------------------------------------------------

def build_graph(classifier: Optional[Any] = None,
                agents: Optional[dict] = None,
                checkpointer: Optional[Any] = None,
                handoff_service: Optional[Any] = None,
                availability_handler: Optional[Any] = None,
                agent_router: Optional[Any] = None,
                session_chat_history: Optional[Any] = None):
    """构建并编译主控编排图。

    Args:
        classifier: 需实现 `async classify_task(task) -> str`；默认包装 TaskClassifier。
        agents: 业务 agent 映射，键为意图（consultation/recommendation/trial/lesson/
                reschedule/cancel），值为实现 `run_stream(user_input)` 的对象。
        checkpointer: LangGraph checkpointer；默认 InMemorySaver()。
        handoff_service: 人工接管服务（record_input/match_keyword/record_kickback/
                trigger_handoff/handoff_reply/handed_off_reply/build_transcript）。
        availability_handler: 档期白名单 handler；默认惰性构造 AvailabilityHandler。
        agent_router: 兜底能力列表路由器（handle_unsupported_task）；默认惰性构造。
        session_chat_history: 会话历史对象（仅用于 transcript 快照，可空）。
    """
    classifier = classifier or TaskClassifierAdapter()
    agents = agents or {}
    checkpointer = checkpointer or InMemorySaver()

    # 缺哪个 agent 就从映射里摘掉对应路由：与 legacy `category == X and agent`
    # 的短路语义一致（无 agent → 落兜底），且保持路由纯函数（路由只读 state）。
    present_business = [node for node in _BUSINESS_NODES
                        if agents.get(node.replace("_node", "")) is not None]
    intent_routes = {intent: node for intent, node in _INTENT_ROUTES.items()
                     if node not in _BUSINESS_NODES or node in present_business}
    resume_routes = {flow: node for flow, node in _RESUME_ROUTES.items()
                     if node not in _BUSINESS_NODES or node in present_business}

    graph = StateGraph(TutorState)
    graph.add_node("entry_node", partial(
        _nodes.entry_node, handoff_service=handoff_service,
        session_chat_history=session_chat_history,
    ))
    graph.add_node("resume_node", _nodes.resume_node)
    graph.add_node("availability_node", partial(
        _nodes.availability_node, handler=availability_handler,
    ))
    graph.add_node("classify_node", partial(_nodes.classify_node, classifier=classifier))
    graph.add_node("verification_node", _nodes.verification_node)
    graph.add_node("busy_node", _nodes.busy_node)
    graph.add_node("error_node", _nodes.error_node)
    graph.add_node("handoff_node", partial(_nodes.handoff_node, handoff_service=handoff_service))
    graph.add_node("handoff_wait_node", partial(
        _nodes.handoff_wait_node, handoff_service=handoff_service))
    graph.add_node("unrelated_node", partial(
        _nodes.unrelated_node, handoff_service=handoff_service,
        agent_router=agent_router, session_chat_history=session_chat_history,
    ))
    for node in present_business:
        graph.add_node(node, partial(
            getattr(_nodes, node), agent=agents[node.replace("_node", "")],
        ))

    business_targets = {node: node for node in present_business}

    graph.add_edge(START, "entry_node")
    graph.add_conditional_edges("entry_node", route_entry, {
        "handoff_node": "handoff_node",
        "verification_node": "verification_node",
        "availability_node": "availability_node",
        "resume_node": "resume_node",
        "classify_node": "classify_node",
        "error_node": "error_node",
    })
    graph.add_conditional_edges("resume_node", partial(
        route_resume, resume_routes=resume_routes), {
        **business_targets, _FALLBACK_NODE: _FALLBACK_NODE,
        "error_node": "error_node",
    })
    graph.add_conditional_edges("classify_node", partial(
        route_after_classify, intent_routes=intent_routes), {
        "busy_node": "busy_node",
        "verification_node": "verification_node",
        **business_targets, _FALLBACK_NODE: _FALLBACK_NODE,
        "error_node": "error_node",
    })

    # 转人工：先出文案（handoff_node），再 interrupt 挂起（handoff_wait_node）。
    graph.add_edge("handoff_node", "handoff_wait_node")
    graph.add_conditional_edges("unrelated_node", route_after_unrelated, {
        "handoff_wait_node": "handoff_wait_node",
        END: END,
    })

    for node in ("handoff_wait_node", "verification_node", "busy_node", "error_node",
                 _FALLBACK_NODE, *present_business):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
