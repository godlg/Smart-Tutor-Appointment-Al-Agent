"""LangGraph 主控编排：节点实现（P0-A：图承载完整主流程）。

设计原则（与 legacy 等价，但不重写业务）：
- 节点**只做转发/判断**，不实现任何业务逻辑（计数、落库、排课、退款仍在 service/agent）。
- 依赖注入：分类器、业务 agent、handoff_service、availability_handler 都在 build_graph
  时注入；节点本身不 new 业务服务，因此可用轻量替身跑通，无需初始化知识库 / LLM / DB。
- 固定文案（`_VERIFICATION_REPLY` / `_BUSY_REPLY` / 分流行 / `[ERROR]` 兜底）一律
  **从原模块导入/复用**（`classification_processor` 常量、`agent_router.route_thought`），
  不复制字符串，避免两处漂移。
- 业务节点调用注入 agent 的 `run_stream(user_input)`，把产出文本追加进 messages；
  同时用 `get_stream_writer()` 逐 token 外发，保持前端 `[THOUGHT]`/`[REPLY]` 流式协议。
- 首轮业务节点先外发与 legacy 逐字一致的分流行（D1）；咨询节点经 `async with agent`
  进入以触发知识库初始化（D3）；各节点包 `_guard_node` 异常兜底（D4）。

节点 ↔ legacy 行为对照（行号以 `classification_processor.py` / `task_classification_agent.py` 为准）：
- entry_node        : A1 `record_input` → A2 已转人工短路 → A3 关键词转人工 → B1 匿名校验
- availability_node : B5 档期白名单直答（不调 LLM、不改 flow_state）
- resume_node       : B2/B3/B4 + route_by_state 续轮（直连，跳过分类器）
- classify_node     : B6 `classify_task`；`ClassificationUnavailable` → busy；B7 匿名校验
- verification_node : B1/B7 匿名校验文案
- busy_node         : B6 降级文案
- handoff_node      : A2/A3 转人工文案
- unrelated_node    : D 踢回阈值 → 转人工；未达阈值 → 能力列表
- consultation/recommendation/trial/lesson/reschedule/cancel_node : C 业务转发
"""

import functools
import logging
from typing import Any, Optional

from config.constants import StateEnum
from config.identity import get_current_identity

from langgraph.errors import GraphBubbleUp

from ..context.types import ContextBudgetExceeded
from ..task_classification.agent_router import route_thought
from ..task_classification.classification_processor import (
    _BUSINESS_CATEGORIES,
    _BUSINESS_STATES,
    _BUSY_REPLY,
    _ERROR_BUDGET_REPLY,
    _ERROR_GENERIC_TEMPLATE,
    _VERIFICATION_REPLY,
)
from ..task_classification.task_classifier import ClassificationUnavailable

from .state import TutorState

logger = logging.getLogger(__name__)

# 客服 `resolve` 结束挂起 thread 时用的恢复哨兵值：`Command(resume=HANDOFF_RESOLVED)`
# 恢复后节点据此判定"已处理，结束挂起"，不产出客服回复文案，让会话回到普通 AI 对话。
HANDOFF_RESOLVED = "__handoff_resolved__"


def last_human_text(state: TutorState) -> str:
    """取最近一条人类消息的纯文本（不查 DB / 不调 LLM）。"""
    for msg in reversed(state.get("messages") or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role in ("human", "user"):
            return content or ""
    return ""


def emit(token: str) -> None:
    """把 token 外发给 LangGraph 的 custom 流；非流式上下文下静默跳过。

    这样同一个节点既能被 astream(stream_mode="custom") 逐 token 消费，
    也能被 ainvoke() 直接调用（此时没有 writer，只落 messages）。
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer(token)
    except Exception:
        logger.debug("stream writer 外发失败(已忽略)", exc_info=True)


def _emit_whole(text: str) -> dict:
    """整段外发（legacy 里 `yield <常量>` 的单 token 形态）。"""
    emit(text)
    return {"messages": [{"role": "ai", "content": text}]}


# LangGraph 内部控制流异常（interrupt / 父 Command 冒泡）不是业务错误，
# 必须原样放行，否则会把人工接管打断。仅拦截真正的业务/运行时异常。
_GUARDED_EXCEPTIONS = (GraphBubbleUp,)


def _fallback(text: str) -> dict:
    """异常兜底返回：外发 `[ERROR]` 文案，置 `error=True` 终止本轮路由。

    legacy `process_task_stream` 的 `except Exception` 会调用
    `self.state_manager.force_reset()`；图里等价语义即把 flow_state 置回 classify，
    并由路由函数据 `error` 短路到 `error_node`（不再进下游业务/兜底节点）。
    """
    emit(text)
    return {"messages": [{"role": "ai", "content": text}],
            "error": True,
            "flow_state": StateEnum.CLASSIFY.value}


def _guard_node(func):
    """给节点加异常兜底包装（等价 legacy `process_task_stream:169-173`）。

    - `ContextBudgetExceeded` → `_ERROR_BUDGET_REPLY`（文案逐字来自原模块，不复制）
    - 其它 `Exception` → `_ERROR_GENERIC_TEMPLATE` + flow_state 复位
    - `GraphBubbleUp`（interrupt 等控制流）原样透传
    """
    @functools.wraps(func)
    async def wrapper(state, *args, **kwargs):
        try:
            return await func(state, *args, **kwargs)
        except _GUARDED_EXCEPTIONS:
            raise
        except ContextBudgetExceeded:
            return _fallback(_ERROR_BUDGET_REPLY)
        except Exception as exc:  # noqa: BLE001 - 与 legacy 兜底口径一致
            return _fallback(_ERROR_GENERIC_TEMPLATE.format(error=exc))

    return wrapper


@_guard_node
async def error_node(state: TutorState) -> dict:
    """D4 兜底终点：异常兜底文案已由 `_guard_node` 外发，这里只终止本轮。"""
    return {}


def _requires_business_verification(flow_state=None, category=None) -> bool:
    """匿名校验判定（与 legacy `ClassificationProcessor._requires_business_verification` 同义）。

    identity 为 None / legacy → False（内部 demo / 单用户路径保持可用）；
    否则命中业务类别或业务态 → True。
    """
    identity = get_current_identity()
    if identity is None or identity.legacy:
        return False
    return category in _BUSINESS_CATEGORIES or flow_state in _BUSINESS_STATES


def _build_transcript(handoff_service, session_chat_history, session_id):
    """会话历史快照 + 最近无关轮输入兜底（与 legacy `_build_transcript` 同义）。"""
    messages = getattr(session_chat_history, "messages", None)
    return handoff_service.build_transcript(session_id, messages)


def _safe_handoff_record(handoff_service, session_id):
    """安全读取该 session 的 handoff 记录（供 interrupt payload / 幂等触发取值）。"""
    getter = getattr(handoff_service, "get_by_session", None)
    if getter is None:
        return None
    try:
        return getter(session_id)
    except Exception:
        return None


def _reset_flags() -> dict:
    """每轮请求复位瞬态判定字段（checkpointer 会跨轮持久，必须显式清零）。"""
    return {
        "handoff": False,
        "error": False,
        "verification_required": False,
        "busy": False,
        "reply_flags": [],
    }


def _from_classify(state: TutorState) -> bool:
    """本轮是否从分类入口进入（flow_state 为 classify / 空）。

    续轮态（flow_state != classify）直连业务节点时，legacy 不会再次 transition，
    图里必须保持一致，否则会把 TRIAL_CONFIRM / TRIAL_CONVERT / LESSON 续接冲掉。
    """
    flow_state = state.get("flow_state")
    return not flow_state or flow_state == StateEnum.CLASSIFY.value


def _resolve_flow_state(agent, default: str) -> str:
    """读业务 agent 当前共享状态作为 flow_state；取不到时用该路由的 legacy 默认值。

    legacy 里 trial/lesson 在路由前显式 transition；consultation/recommendation/
    reschedule/cancel 不 transition。业务 agent 内部若自行推进状态（如 TrialAgent
    切 TRIAL_CONFIRM / TRIAL_CONVERT / RecommendationAgent 切 RECOMMENDATION_CONFIRM），
    这里以 agent 实际状态为准（含 shared_state 别名）。
    """
    agent_state = getattr(agent, "state", None)
    if agent_state is None:
        agent_state = getattr(agent, "shared_state", None)
    value = getattr(agent_state, "value", None)
    if value is None:
        return default
    return getattr(value, "value", value)


# ----------------------------------------------------------------------
# 入口层（A1 → A2 → A3 → B1）
# ----------------------------------------------------------------------

@_guard_node
async def entry_node(state: TutorState, *, handoff_service=None,
                     session_chat_history=None) -> dict:
    """入口：记录输入、已转人工短路、主动关键词转人工、匿名校验判定。

    纯转发 handoff_service：不在图里实现计数 / 落库 / 文案。
    """
    out = _reset_flags()
    task = last_human_text(state)
    if handoff_service is None:
        return out

    session_id = state.get("session_id") or "default"
    # A1：记录用户输入（含无关轮，供 transcript 兜底）
    handoff_service.record_input(session_id, task)

    # A2：已转人工 → 家长消息落库 + 短路（不回分类）
    if handoff_service.is_handed_off(session_id):
        try:
            handoff_service.parent_message(session_id, task)
        except Exception as e:
            logger.warning("记录家长消息失败(已忽略): %s", e)
        out["handoff"] = True
        out["reply_flags"] = ["handed_off"]
        return out

    # A3：主动转人工关键词 → 触发转人工 + 触发文案
    kw = handoff_service.match_keyword(task)
    if kw:
        handoff_service.trigger_handoff(
            session_id, "keyword", reason=kw,
            transcript=_build_transcript(handoff_service, session_chat_history, session_id),
        )
        out["handoff"] = True
        out["reply_flags"] = ["handoff"]
        return out

    # B1：匿名 + 业务态 → 需校验（在续轮态之前判定，与 legacy 顺序一致）
    if _requires_business_verification(flow_state=state.get("flow_state")):
        out["verification_required"] = True
    return out


# ----------------------------------------------------------------------
# 分类前：档期白名单（B5）、续轮态（B2/B3/B4 决策锚点）
# ----------------------------------------------------------------------

@_guard_node
async def availability_node(state: TutorState, *, handler=None) -> dict:
    """B5：全局档期查询白名单 → 规则直答（不调 LLM、不改 flow_state）。"""
    task = last_human_text(state)
    if handler is None:
        from ..availability.availability_handler import AvailabilityHandler

        handler = AvailabilityHandler()
    buf = ""
    async for token in handler.handle(task):
        buf += token
        emit(token)
    # 刻意不返回 flow_state：白名单不改状态机
    return {"messages": [{"role": "ai", "content": buf}]}


async def resume_node(state: TutorState) -> dict:
    """B2/B3/B4 决策锚点：本身不做事，路由交给纯函数 route_resume 决定目标节点。"""
    return {}


# ----------------------------------------------------------------------
# 分类（B6 / B7）
# ----------------------------------------------------------------------

@_guard_node
async def classify_node(state: TutorState, classifier) -> dict:
    """B6：调用注入分类器写回 intent；`ClassificationUnavailable` → busy 降级。

    顺带做 B7：分类后匿名 + 业务类别 → 置 verification_required（保持在 state 里，
    让 route_after_classify 保持纯函数）。
    """
    task = last_human_text(state)
    try:
        intent = await classifier.classify_task(task)
    except ClassificationUnavailable:
        # 不踢回 / 不转人工 / 不改状态，交给 busy_node 回降级文案
        return {"busy": True}
    emit(f"[THOUGHT][LangGraph] 意图识别为 {intent}。")
    out = {"intent": intent, "flow_state": StateEnum.CLASSIFY.value}
    if _requires_business_verification(category=intent):
        out["verification_required"] = True
    return out


# ----------------------------------------------------------------------
# 短路节点：校验 / 繁忙 / 转人工
# ----------------------------------------------------------------------

async def verification_node(state: TutorState) -> dict:
    """B1/B7：匿名访问业务数据 → 校验引导文案（与 legacy 逐字一致）。"""
    return _emit_whole(_VERIFICATION_REPLY)


async def busy_node(state: TutorState) -> dict:
    """B6 降级：分类服务不可用 → 系统繁忙文案（与 legacy 逐字一致）。"""
    return _emit_whole(_BUSY_REPLY)


def _handoff_interrupt_payload(handoff_service, session_id, trigger_type, record=None):
    """构造 `interrupt()` payload（至少含 handoff_id / trigger_type / reason / session_id）。

    trigger_type 取 DB 记录为准（幂等记录可能是更早的关键词/阈值触发）；取不到时用调用方
    传入值。handoff_service / 记录缺失时返回 None（不挂起，保持纯文案路径，图可离线跑）。
    """
    if handoff_service is None:
        return None
    if record is None and hasattr(handoff_service, "get_by_session"):
        try:
            record = handoff_service.get_by_session(session_id)
        except Exception:
            record = None
    if not record:
        return None
    return {
        "handoff_id": record.get("id"),
        "trigger_type": record.get("trigger_type") or trigger_type,
        "reason": record.get("trigger_reason"),
        "session_id": record.get("session_id") or session_id,
    }


@_guard_node
async def handoff_node(state: TutorState, *, handoff_service=None) -> dict:
    """A2/A3：转人工文案（先外发协议标记，再逐字外发文案）。

    文案来源优先注入的 handoff_service（单一实现），否则从服务层常量导入。
    本节点只负责"文案 + 落库已完成（entry_node）"；随后的 `handoff_wait_node` 才
    `interrupt()` 挂起。两节点拆分是为了：恢复（`Command(resume=...)`）时只重跑挂起
    节点，**不会重复外发/重复落库本节点文案**（幂等）。
    """
    flags = state.get("reply_flags") or []
    if "handed_off" in flags:
        prefix = "[REPLY]"
        text = (handoff_service.handed_off_reply() if handoff_service
                else _handed_off_reply_const())
    else:
        prefix = "[REPLY][HANDOFF]"
        text = (handoff_service.handoff_reply() if handoff_service
                else _handoff_reply_const())
    emit(prefix)
    for char in text:
        emit(char)
    return {"messages": [{"role": "ai", "content": prefix + text}], "handoff": True}


@_guard_node
async def handoff_wait_node(state: TutorState, *, handoff_service=None) -> dict:
    """P2-1：用 `interrupt()` 挂起转人工会话，等客服 `Command(resume=...)`。

    payload 至少含 `handoff_id` / `trigger_type` / `reason` / `session_id`。落库已在
    entry_node / unrelated_node 完成且早于本节点，因此不会被 interrupt 的异常式中断吞掉。

    resume 语义：
    - 客服回复正文 → 外发 `[REPLY]<content>` 并落进 messages（供客服/前端看到回复）；
    - `HANDOFF_RESOLVED`（resolve 哨兵）→ 结束挂起，置 `handoff=False`，回到普通 AI 对话。
    """
    session_id = state.get("session_id") or "default"
    record = _safe_handoff_record(handoff_service, session_id)
    payload = _handoff_interrupt_payload(handoff_service, session_id, "keyword",
                                         record=record)
    if payload is None:
        # 无 handoff_service / 无记录（离线替身）：不挂起，保持文案路径可跑通。
        return {}

    resume_value = interrupt_handoff(payload)
    if resume_value == HANDOFF_RESOLVED:
        return {"handoff": False}
    if resume_value is None:
        return {}
    reply_text = f"[REPLY]{resume_value}"
    emit(reply_text)
    return {"messages": [{"role": "ai", "content": reply_text}]}


def _handoff_reply_const() -> str:
    from services.human_handoff_service import HANDOFF_REPLY

    return HANDOFF_REPLY


def _handed_off_reply_const() -> str:
    from services.human_handoff_service import HANDED_OFF_REPLY

    return HANDED_OFF_REPLY


# ----------------------------------------------------------------------
# 业务节点（C）：只转发注入 agent 的 run_stream
# ----------------------------------------------------------------------

def _transition(agent, flow_state: str) -> None:
    """路由前显式推进 agent 共享状态（mirror legacy transition_to_trial / _lesson）。

    仅在 agent 暴露可写共享状态时生效；替身无 state 时静默跳过。
    """
    shared = getattr(agent, "state", None)
    if shared is None:
        shared = getattr(agent, "shared_state", None)
    if shared is None:
        return
    try:
        from agents.task_classification.state_manager import FlowState

        shared.value = FlowState(flow_state)
    except Exception:
        pass


def _accepts_intent(callable_obj) -> bool:
    """判断 run_stream 是否接收 intent 关键字（真 LessonAgent 接收，轻量替身可能不接收）。"""
    import inspect

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return "intent" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _supports_aenter(agent) -> bool:
    """agent 是否可作异步上下文管理器（ConsultantAgent 借此触发知识库初始化）。"""
    if agent is None:
        return False
    return callable(getattr(type(agent), "__aenter__", None)) and \
        callable(getattr(type(agent), "__aexit__", None))


def _agent_stream(agent, task: str, intent: Optional[str]):
    """按 agent.run_stream 是否接收 intent 选调用形态。"""
    if intent and _accepts_intent(agent.run_stream):
        return agent.run_stream(user_input=task, intent=intent)
    return agent.run_stream(user_input=task)


async def _forward_agent(state: TutorState, agent, default_flow_state: str,
                         intent: Optional[str] = None,
                         transition_to: Optional[str] = None,
                         thought_intent: Optional[str] = None,
                         enter_agent: bool = False) -> dict:
    """业务节点公共转发逻辑：只调用 agent.run_stream，不做业务判断。

    仅做四件编排性的事：
    1. 首轮分类命中业务意图时，先外发与 legacy **逐字一致**的分流行
       （`route_thought`，与 `AgentRouter.route_to_*` 同源，不复制字符串）；
    2. 路由前按 legacy 语义显式 transition（trial/lesson 需要）；
    3. 咨询节点经 `async with agent` 进入（等价 legacy `route_to_consultation`
       的 `async with self.consultant_agent`，触发 `__aenter__` → 知识库初始化）；
    4. 逐 token 外发，结束后把 agent 共享状态写回 flow_state 供下一轮续接。
    """
    task = last_human_text(state)
    if agent is None:
        reply = f"[ERROR]{default_flow_state} 服务暂时不可用"
        emit(reply)
        return {"messages": [{"role": "ai", "content": reply}]}

    if transition_to:
        _transition(agent, transition_to)

    # D1：首轮分流行（含前缀），复用 AgentRouter 的文案；续轮态不产生。
    buf = ""
    if thought_intent:
        buf = route_thought(thought_intent)
        emit(buf)

    if enter_agent:
        async with agent as entered:
            async for token in _agent_stream(entered, task, intent):
                buf += token
                emit(token)
    else:
        async for token in _agent_stream(agent, task, intent):
            buf += token
            emit(token)
    return {
        "messages": [{"role": "ai", "content": buf}],
        "flow_state": _resolve_flow_state(agent, default_flow_state),
    }


@_guard_node
async def consultation_node(state: TutorState, agent=None) -> dict:
    # legacy：咨询路由不显式 transition（ConsultantAgent 完成后自行置回 CLASSIFY）。
    # D3：经 `async with agent` 进入，等价 legacy `route_to_consultation` 的
    # `async with self.consultant_agent`（触发 __aenter__ → knowledge_retriever.initialize）。
    # D1：仅首轮（从 classify 进入）产生分流行；consultation 续轮态（route_by_state）无此行。
    return await _forward_agent(
        state, agent, StateEnum.CLASSIFY.value,
        thought_intent="consultation" if _from_classify(state) else None,
        enter_agent=_supports_aenter(agent),
    )


@_guard_node
async def recommendation_node(state: TutorState, agent=None) -> dict:
    # legacy：推荐路由不显式 transition（RecommendationAgent 确认态时自行置 RECOMMENDATION_CONFIRM）。
    # D1：仅首轮从 classify 进入时产生分流行；RECOMMENDATION_CONFIRM 续轮（route_to_recommendation_confirm）无。
    return await _forward_agent(
        state, agent, StateEnum.CLASSIFY.value,
        thought_intent="recommendation" if _from_classify(state) else None,
    )


@_guard_node
async def trial_node(state: TutorState, agent=None) -> dict:
    """试讲节点：命中『转人工』时用 interrupt() 演示人工接管，否则正常转发。

    transition 语义与 legacy 对齐：仅从分类入口（flow_state=classify）进入时才
    `transition_to_trial`；续轮态（TRIAL_CONFIRM / TRIAL_CONVERT）直连时不改状态，
    否则会把待确认/待转化态冲掉。
    """
    if "转人工" in last_human_text(state):
        answer = interrupt_handoff({"reason": "human_handoff"})
        text = f"[REPLY][人工接管] 已为您转接人工客服。恢复输入：{answer}"
        emit(text)
        return {
            "messages": [{"role": "ai", "content": text}],
            "handoff": True,
            "flow_state": StateEnum.TRIAL.value,
        }
    # legacy：category=="trial" 路由前显式 transition_to_trial；续轮直连不 transition。
    from_classify = _from_classify(state)
    transition = StateEnum.TRIAL.value if from_classify else None
    # D1：首轮分流行；TRIAL_CONFIRM / TRIAL_CONVERT 续轮（route_to_trial_confirm/convert）无。
    return await _forward_agent(state, agent, StateEnum.TRIAL.value,
                                transition_to=transition,
                                thought_intent="trial" if from_classify else None)


@_guard_node
async def lesson_node(state: TutorState, agent=None) -> dict:
    # legacy：category=="lesson" 路由前显式 transition_to_lesson；续轮直连不 transition。
    from_classify = _from_classify(state)
    transition = StateEnum.LESSON.value if from_classify else None
    # D1：首轮分流行；LESSON 续轮不产生。
    return await _forward_agent(state, agent, StateEnum.LESSON.value,
                                transition_to=transition,
                                thought_intent="lesson" if from_classify else None)


@_guard_node
async def reschedule_node(state: TutorState, agent=None) -> dict:
    # legacy：route_to_reschedule → lesson_agent.run_stream(intent="reschedule")，带分流行。
    return await _forward_agent(state, agent, StateEnum.CLASSIFY.value,
                                intent="reschedule", thought_intent="reschedule")


@_guard_node
async def cancel_node(state: TutorState, agent=None) -> dict:
    # legacy：route_to_cancel → lesson_agent.run_stream(intent="cancel")，带分流行。
    return await _forward_agent(state, agent, StateEnum.CLASSIFY.value,
                                intent="cancel", thought_intent="cancel")


def interrupt_handoff(payload: dict):
    """薄封装 langgraph.types.interrupt（单独抽出便于替换 / 单测）。"""
    from langgraph.types import interrupt

    return interrupt(payload)


# ----------------------------------------------------------------------
# 兜底（D）
# ----------------------------------------------------------------------

@_guard_node
async def unrelated_node(state: TutorState, *, handoff_service=None,
                         agent_router=None, session_chat_history=None) -> dict:
    """D：踢回计数达阈值 → 转人工；未达阈值 → 能力列表。"""
    session_id = state.get("session_id") or "default"
    if handoff_service is not None and handoff_service.record_kickback(session_id):
        count = handoff_service.get_count(session_id)
        handoff_service.trigger_handoff(
            session_id, "kickback_threshold",
            reason=f"累计踢回 {count} 次",
            unrelated_count=count,
            transcript=_build_transcript(handoff_service, session_chat_history, session_id),
        )
        prefix = "[REPLY][HANDOFF]"
        emit(prefix)
        text = handoff_service.handoff_reply()
        for char in text:
            emit(char)
        # 落库（trigger_handoff）已完成；挂起由后续 handoff_wait_node 的 interrupt() 承担
        # （路由按 `handoff=True` 把本轮导到 handoff_wait_node）。
        return {
            "messages": [{"role": "ai", "content": prefix + text}],
            "handoff": True,
            "unrelated_count": count,
            "flow_state": StateEnum.UNRELATED.value,
        }

    router = agent_router
    if router is None:
        from ..task_classification.agent_router import AgentRouter

        router = AgentRouter()
    buf = ""
    async for token in router.handle_unsupported_task(state.get("intent") or ""):
        buf += token
        emit(token)
    return {
        "messages": [{"role": "ai", "content": buf}],
        "flow_state": StateEnum.UNRELATED.value,
    }
