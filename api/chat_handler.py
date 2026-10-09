"""
聊天处理器（家教场景）

会话隔离: 同一个 session_id 复用同一个 TaskClassificationAgent,
不同 session 走不同的 StateManager(避免多家长/多窗口互相覆盖)。
session_id 没传时退回全局默认 agent(向后兼容 demo / 单用户场景)。
"""

import os
import uuid
import logging
import asyncio
import threading
import time
from typing import Any, AsyncGenerator, Dict, Optional

from config.identity import Identity
from config.memory import SESSION_CACHE_TTL_SECONDS, SESSION_CACHE_MAX_ENTRIES

from agents.task_classification_agent import TaskClassificationAgent
from agents.trial import TrialAgent
from agents.lesson import LessonAgent
from agents.consultant_agent import ConsultantAgent
from agents.recommendation_agent import RecommendationAgent
from services.conversation_service import strip_thought_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 会话历史服务(统一写入 / 恢复)。懒加载,便于测试注入临时库。
# ---------------------------------------------------------------------------

_conversation_service = None


def get_conversation_service():
    """懒加载会话历史服务(单例)。测试可 monkeypatch api.chat_handler._conversation_service。"""
    global _conversation_service
    if _conversation_service is None:
        from services.conversation_service import ConversationService
        _conversation_service = ConversationService()
    return _conversation_service


# ---------------------------------------------------------------------------
# 默认(单租户) agent
# ---------------------------------------------------------------------------

_global_session_id: str = str(uuid.uuid4())
_global_lock = threading.Lock()
_task_agent: Optional[TaskClassificationAgent] = None


def _build_task_agent(session_id: str, identity: Optional[Identity] = None) -> TaskClassificationAgent:
    from agents.context.runtime import build_runtime_context
    from config.identity import legacy_identity
    from services.human_handoff_service import HumanHandoffService

    conversation = get_conversation_service()
    identity = conversation.ensure_session(identity or legacy_identity(session_id))
    history = conversation.restore_history(identity)
    _session_chat_histories[session_id] = history

    def runtime():
        return build_runtime_context(identity, history, conversation)

    # 业务 Agent 只读取可见历史，字段提取 JSON 留在各自局部历史。
    agent = TaskClassificationAgent(
        consultant_agent=ConsultantAgent(
            session_id=session_id, session_chat_history=history,
            identity=identity, conversation_service=conversation,
        ),
        trial_agent=TrialAgent(session_id=session_id, session_chat_history=history,
                               context_manager=runtime()),
        lesson_agent=LessonAgent(session_id=session_id, context_manager=runtime()),
        recommendation_agent=RecommendationAgent(
            session_id=session_id, session_chat_history=history, context_manager=runtime()),
        context_manager=runtime(), session_id=session_id,
        session_chat_history=history,
        handoff_service=HumanHandoffService(
            db_path=str(conversation.db_router.session_manager.engine.url)),
    )
    agent.identity = identity
    return agent


def _get_default_task_agent() -> TaskClassificationAgent:
    """懒加载全局默认 agent(单租户 / 旧调用方使用)。"""
    global _task_agent
    if _task_agent is None:
        with _global_lock:
            if _task_agent is None:
                _task_agent = _build_task_agent(_global_session_id)
    return _task_agent


def reload_agents() -> TaskClassificationAgent:
    """支持运行时刷新默认 agent(供测试 / 配置变更)。"""
    global _task_agent
    with _global_lock:
        _task_agent = _build_task_agent(_global_session_id)
    return _task_agent


# ---------------------------------------------------------------------------
# 按 session 缓存 agent
# ---------------------------------------------------------------------------

_session_agents: Dict[str, TaskClassificationAgent] = {}
# 注意:get_task_agent 持锁期间会调 _build_task_agent → get_session_chat_history,
# 两者共用同一把锁。必须用 RLock(可重入),普通 Lock 会同线程自锁死。
_session_lock = threading.RLock()

# 跨 Agent 共享的 session 级聊天历史:RecommendationAgent / ConsultantAgent /
# TrialAgent / LessonAgent 每次写出 AI 回复时都同步写入对应 session 的历史,
# 下游 Agent(尤其 TrialAgent)从这份历史读"上游已确认上下文",避免短输入(如
# 『试讲』)因字段未知被误判 unrelated 踢回主路由。按 session_id 懒创建并复用。
_session_chat_histories: Dict[str, "InMemoryChatMessageHistory"] = {}
_session_last_access: Dict[str, float] = {}


def get_session_chat_history(session_id: str) -> "InMemoryChatMessageHistory":
    """按 session 拿(懒创建)一个跨 Agent 共享的 chat history。

    同一 session_id 永远返回同一对象;不同 session 各自独立,互不覆盖。
    返回类型为 langchain_core 的 InMemoryChatMessageHistory(运行时导入,
    避免顶层引入拖慢冷启动)。
    """
    with _session_lock:
        hist = get_conversation_service().get_history(session_id)
        _session_chat_histories[session_id] = hist
        _session_last_access[session_id] = time.monotonic()
        return hist


def get_task_agent(session_id: Optional[str] = None,
                   identity: Optional[Identity] = None) -> TaskClassificationAgent:
    """
    根据 session_id 获取(或懒构建)一个 TaskClassificationAgent。

    同一 session_id 多次调用返回同一对象,
    不同 session_id 各自有独立的 StateManager / chat history,
    避免多家长或多窗口同时使用时状态互相覆盖。
    """
    from config.identity import get_current_identity
    identity = identity or get_current_identity()
    session_id = session_id or getattr(identity, "session_id", None)
    if not session_id:
        return _get_default_task_agent()
    if identity is not None:
        identity = get_conversation_service().ensure_session(identity.with_session(session_id))
    with _session_lock:
        agent = _session_agents.get(session_id)
        if agent is None:
            agent = _build_task_agent(session_id, identity=identity)
            _session_agents[session_id] = agent
        _session_last_access[session_id] = time.monotonic()
        return agent


def drop_session_agent(session_id: str) -> None:
    """显式释放某个 session 的 agent(可选,目前用于测试)。"""
    with _session_lock, _async_locks_guard:
        if _active_sessions.get(session_id, 0):
            return
        _session_agents.pop(session_id, None)


# ---------------------------------------------------------------------------
# 每 session 一把 async 锁:同一 session 输入串行,不同 session 并行
# ---------------------------------------------------------------------------

_session_async_locks: Dict[str, "asyncio.Lock"] = {}
_active_sessions: Dict[str, int] = {}   # session_id -> 在途请求数(缓存淘汰不淘汰在途会话)
_async_locks_guard = threading.Lock()


def _get_session_lock(session_id: str) -> "asyncio.Lock":
    with _async_locks_guard:
        lock = _session_async_locks.get(session_id)
        if lock is None:
            import asyncio as _asyncio
            lock = _asyncio.Lock()
            _session_async_locks[session_id] = lock
        return lock


def drop_session_lock(session_id: str) -> None:
    """清理 session 的 async 锁(仅在无在途请求时,避免丢弃使用中的锁)。

    持久记录仍保留;后续可重建。
    """
    with _async_locks_guard:
        if _active_sessions.get(session_id, 0) > 0:
            return  # 正在运行,不淘汰
        _session_async_locks.pop(session_id, None)


def drop_session(session_id: str) -> None:
    """清理某 session 的全部内存态(agent / 共享历史 / 锁)。持久记录仍保留。"""
    with _session_lock, _async_locks_guard:
        if _active_sessions.get(session_id, 0):
            return
        _session_agents.pop(session_id, None)
        _session_chat_histories.pop(session_id, None)
        _session_async_locks.pop(session_id, None)
        _session_last_access.pop(session_id, None)
        if _conversation_service is not None:
            _conversation_service.drop_history(session_id)


def evict_idle_sessions(now: Optional[float] = None) -> None:
    """淘汰超时/超容量的闲置会话；运行和排队请求均受 active 保护。"""
    now = time.monotonic() if now is None else now
    with _session_lock:
        ordered = sorted(_session_last_access, key=_session_last_access.get)
        for session_id in ordered:
            expired = now - _session_last_access.get(session_id, now) >= SESSION_CACHE_TTL_SECONDS
            over_limit = len(_session_last_access) > SESSION_CACHE_MAX_ENTRIES
            if expired or over_limit:
                drop_session(session_id)


# 向后兼容的模块级单例(老代码 `from api.chat_handler import task_agent`)
# 用 property 的方式动态获取,确保第一个访问的人触发懒加载。
def __getattr__(name: str):
    if name == "task_agent":
        return _get_default_task_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

async def ProcessUserInput_stream(
    user_input: str,
    state: Optional[str] = None,
    context: Optional[dict] = None,
    session_id: Optional[str] = None,
    identity: Optional["Identity"] = None,
) -> AsyncGenerator[str, None]:
    """
    统一入口：处理用户输入流。

    记忆可靠性 Phase 1 接线:
    - 身份:优先用显式 identity 参数;否则读请求级 ContextVar(config.identity);
      再否则按 session_id 构造 legacy 身份(向后兼容 demo/内部路径)。
    - 统一写入:开始记录 human;完整回复结束过滤 [THOUGHT] 后记录 ai;
      取消/异常时 human 保留、ai 只保存已产生的可用内容(partial),不伪造完整问答。
    - 并发:同一 session 输入按 async 锁串行,不同 session 并行;锁生命周期异常也释放。
    - 恢复:首次进入某 session 时从持久库恢复历史(重复恢复不重复消息)。

    Args:
        user_input: 家长/学员输入
        state: 兼容旧接口,目前未使用
        context: 兼容旧接口,目前未使用
        session_id: 会话 id(为空则走全局单例)
        identity: 可注入身份(不传则从 ContextVar / legacy 兜底)
    """
    from config.identity import set_current_identity, reset_current_identity

    conversation = get_conversation_service()
    ident = conversation.ensure_session(_resolve_entry_identity(session_id, identity))
    stream = _process_session_stream(user_input, conversation, ident)
    try:
        while True:
            # ContextVar token 只覆盖一次推进，不能跨 yield 留在调用方任务中。
            identity_token = set_current_identity(ident)
            try:
                part = await anext(stream)
            except StopAsyncIteration:
                break
            finally:
                reset_current_identity(identity_token)
            yield part
    finally:
        # aclose 可以来自另一个 task；在关闭者自己的 Context 中配对 set/reset。
        identity_token = set_current_identity(ident)
        try:
            await stream.aclose()
        finally:
            reset_current_identity(identity_token)


async def _process_session_stream(user_input: str, conversation, ident: Identity) -> AsyncGenerator[str, None]:
    from contextlib import aclosing

    # P3：默认走 LangGraph 图编排；`ORCHESTRATOR=legacy` 显式回滚旧链路。
    # 图路径在「会话不具备图依赖」时自身降级回 legacy（见 _process_graph_stream）。
    if _env_orchestrator() == "graph":
        async with aclosing(_process_graph_stream(user_input, conversation, ident)) as stream:
            async for token in stream:
                yield token
        return

    async with aclosing(_process_legacy_stream(user_input, conversation, ident)) as stream:
        async for token in stream:
            yield token


async def _process_legacy_stream(user_input: str, conversation, ident: Identity) -> AsyncGenerator[str, None]:
    """legacy 主链路（原 _process_session_stream 函数体，行为逐字未改）。

    P3 保留为回滚路径：`ORCHESTRATOR=legacy` 直接走本函数；图模式下若会话不具备
    图依赖（会话级 agent 非真实 `TaskClassificationAgent`）也降级到本函数。
    """
    from contextlib import aclosing
    from agents.context.types import ContextBudgetExceeded

    session_id = ident.session_id
    _enter_session(session_id)
    try:
        evict_idle_sessions()
        async with _get_session_lock(session_id):
            conversation.record_human(ident, user_input)
            buf = ""
            partial = True
            agent = None
            is_checkpoint_ready = False
            try:
                agent = get_task_agent(session_id=session_id, identity=ident)
                notice = _checkpoint_notice(agent, conversation, ident, user_input)
                is_checkpoint_ready = True
                if notice:
                    buf = "[REPLY]" + notice
                    yield buf
                else:
                    async with aclosing(agent.classify_task_stream(user_input)) as response:
                        async for token in response:
                            buf += token
                            yield token
                partial = False
            except ContextBudgetExceeded:
                # 明确的用户可处理拒绝；不删除当前问题、也不伪造完整答案。
                token = "[ERROR]上下文超过安全预算，请缩短本次输入后重试。"
                buf += token
                yield token
            finally:
                # 收尾仍在同一会话锁内，异常/取消也先落可用部分，再释放锁。
                _finalize_turn(conversation, ident, buf, partial, agent, is_checkpoint_ready)
    finally:
        _exit_session(session_id)
        evict_idle_sessions()


def _checkpoint_notice(agent, conversation, identity: "Identity", user_input: str) -> str:
    """恢复门闩：重建 agent 首轮仅展示草稿并重新询问，不执行业务。

    两条链路(legacy / graph)共用，避免复制粘贴。返回 notice 文案，无则空串。
    """
    notice = _restore_agent_checkpoint(agent, conversation, identity)
    if not notice and getattr(agent, "_requires_reconfirmation", False):
        from services.session_checkpoint import RECONFIRMATION_NOTICE
        if _has_fresh_checkpoint_requirements(user_input):
            agent._requires_reconfirmation = False
        else:
            notice = RECONFIRMATION_NOTICE
    return notice


def _finalize_turn(conversation, identity: "Identity", buf: str,
                   partial: bool, agent, is_checkpoint_ready: bool) -> None:
    """收尾落库：过滤 [THOUGHT] 后记录 AI 回复 + 保存 checkpoint（两条链路共用）。"""
    cleaned = strip_thought_output(buf)
    if cleaned:
        try:
            conversation.record_ai(identity, cleaned, partial=partial or "[ERROR]" in cleaned)
        except Exception as exc:
            logger.warning("记录 AI 回复失败: %s", type(exc).__name__)
    if is_checkpoint_ready:
        _save_agent_checkpoint(agent, conversation, identity)


# ---------------------------------------------------------------------------
# LangGraph 编排旁路（ORCHESTRATOR=graph 时启用；默认关闭）
# ---------------------------------------------------------------------------

_graph_lock = threading.Lock()
_graph_checkpointer = None
_graph_compiled: Dict[str, Any] = {}
# 编译图绑定的是「构建时那个会话级 agent」；agent 被重建（测试注入 / 会话淘汰后
# 重建）后必须失效重编，否则图会一直跑旧 agent（及其已失效的 KB/DB 连接）。
# 存对象本身（而非 id()）以避免对象被回收后 id 复用导致误判「未变」。
_graph_bound_agent: Dict[str, Any] = {}
# P1 持久化：AsyncSqliteSaver 绑定 running loop，按 loop 缓存实例；
# 同一 loop 复用同一连接（不每轮重连），loop 变化时才重建。
_persistent_savers: Dict[int, Any] = {}
_graph_loop: Optional[Any] = None


def _env_orchestrator() -> str:
    # P3 暂缓：默认仍走 legacy（P3-1 曾切 graph，因图路径会话语义未对齐而回退）；
    # `ORCHESTRATOR=graph` 显式启用 LangGraph 图编排。
    return os.getenv("ORCHESTRATOR", "legacy").lower()


def _graph_thread_id(session_id: str) -> str:
    """以 session_id 作为 thread_id，并加 graph 前缀与 legacy 状态隔离。"""
    return f"graph:{session_id}"


def _get_graph_checkpointer():
    """返回图 checkpointer（默认 InMemorySaver；GRAPH_PERSISTENCE=1 时 AsyncSqliteSaver）。

    - 默认（未开 `GRAPH_PERSISTENCE`）返回进程内 `InMemorySaver` 单例，既有行为不变。
    - 开启后返回 `AsyncSqliteSaver`，落到项目既有 SQLite 库（checkpointer 另建
      `checkpoints` / `writes` 表，与既有业务表同库共存，不改任何既有表）。
      `AsyncSqliteSaver` 构造即绑定 running loop，故这里按 loop 缓存：同一 loop
      复用同一连接（满足「不要每轮重复建连接」），loop 变化时才新建。
    - 无事件循环的同步调用方退回 `InMemorySaver`，不因构造失败而崩溃。
    """
    global _graph_checkpointer, _graph_loop

    from agents.graph import persistence_enabled

    if persistence_enabled():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            with _graph_lock:
                saver = _persistent_savers.get(id(loop))
                if saver is None:
                    from agents.graph import build_checkpointer

                    saver = build_checkpointer(persistent=True)
                    _persistent_savers[id(loop)] = saver
                _graph_loop = loop
                return saver

    if _graph_checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        _graph_checkpointer = InMemorySaver()
    return _graph_checkpointer


def _graph_loop_changed() -> bool:
    """持久化模式下：当前 loop 与缓存 Saver 所属 loop 不同（需失效重建）。

    生产单 loop 下恒为 False；测试多次 `asyncio.run` 时抬手避免复用绑定到
    已关闭 loop 的旧 graph / 旧连接（会 RuntimeError）。
    """
    from agents.graph import persistence_enabled

    if not persistence_enabled() or _graph_loop is None:
        return False
    try:
        return asyncio.get_running_loop() is not _graph_loop
    except RuntimeError:
        return False


def _reset_graph_checkpointers() -> None:
    """清空图缓存与持久化 Saver 实例（测试 / 运行期刷新使用）。

    同步上下文无法 await 关闭 aiosqlite 连接；loop 内调用方可先
    `await close_checkpointer(saver)`，这里仅解除引用，避免跨 loop 复用。
    """
    global _graph_checkpointer, _graph_loop
    with _graph_lock:
        _graph_compiled.clear()
        _graph_bound_agent.clear()
        _persistent_savers.clear()
        _graph_checkpointer = None
        _graph_loop = None


async def _ensure_graph_checkpointer_setup() -> None:
    """持久化 Saver 首次使用前显式建表（`AsyncSqliteSaver.setup` 幂等）。

    InMemorySaver 无 `setup`，此处为无操作，默认路径行为不变。
    """
    saver = _get_graph_checkpointer()
    setup = getattr(saver, "setup", None)
    if setup is not None:
        await setup()


def _build_graph_deps(session_id: str, identity: Optional[Identity] = None):
    """复用 session 级 TaskClassificationAgent 的分类器/业务 agent/服务（只读）。

    复用真实 TaskClassifier 可保证图路径与 legacy 的分类归一完全一致；
    handoff_service / availability_handler / agent_router 同样取自该 agent 的
    classification_processor，使图路径与 legacy 共享同一份会话级计数与文案。
    取不到时对应项为 None，由 build_graph / 节点做惰性兜底。

    identity 显式透传（与 legacy `get_task_agent(session_id, identity=ident)` 一致）：
    图在 astream 的任务里跑，ContextVar 不保证可见，只靠 session_id 会拿到不同
    身份命名空间的会话级 agent（甚至重建），破坏与 legacy 的会话一致性。
    """
    from agents.task_classification.classification_processor import ClassificationProcessor

    agent = get_task_agent(session_id=session_id, identity=identity)
    processor = getattr(agent, "classification_processor", None)
    classifier = None
    deps = {"handoff_service": None, "availability_handler": None,
            "agent_router": None, "session_chat_history": None}
    if isinstance(processor, ClassificationProcessor):
        classifier = getattr(processor, "task_classifier", None)
        deps["handoff_service"] = getattr(processor, "handoff_service", None)
        deps["availability_handler"] = getattr(processor, "availability_handler", None)
        deps["agent_router"] = getattr(processor, "agent_router", None)
        deps["session_chat_history"] = getattr(processor, "session_chat_history", None)
        router = deps["agent_router"]
        if router is not None:
            lesson = getattr(router, "lesson_agent", None)
            return classifier, _normalize_agents({
                "consultation": getattr(router, "consultant_agent", None),
                "recommendation": getattr(router, "recommendation_agent", None),
                "trial": getattr(router, "trial_agent", None),
                "lesson": lesson,
                "reschedule": lesson,
                "cancel": lesson,
            }), deps
    return classifier, _normalize_agents({}), deps


def _normalize_agents(agents: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, agent in agents.items():
        if agent is None:
            continue
        # 业务 agent 的流式入口不统一：run_stream / consult_stream 二者取一。
        if hasattr(agent, "run_stream"):
            out[key] = agent
        elif hasattr(agent, "consult_stream"):
            out[key] = _ConsultStreamAdapter(agent)
    return out


class _ConsultStreamAdapter:
    """把 ConsultantAgent.consult_stream(user_input) 适配成 run_stream(user_input,intent=None)。

    仅做签名适配，不实现业务逻辑；图节点因此对"run_stream 协议"保持一致。
    透传 `__aenter__`/`__aexit__`：图咨询节点会经 `async with agent` 进入，
    等价 legacy `route_to_consultation` 的 `async with consultant_agent`，
    从而触发 ConsultantAgent.__aenter__ → knowledge_retriever.initialize()（D3）。
    """

    def __init__(self, consultant):
        self._consultant = consultant
        # 透出可写共享状态，供图节点读/写 flow_state（与真实 agent 同型）。
        self.state = getattr(consultant, "shared_state", None)
        self.shared_state = self.state

    async def __aenter__(self):
        await self._consultant.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._consultant.__aexit__(exc_type, exc, tb)

    async def run_stream(self, user_input: str, intent=None) -> AsyncGenerator[str, None]:
        from contextlib import aclosing

        async with aclosing(self._consultant.consult_stream(user_input)) as stream:
            async for token in stream:
                yield token


def _build_orchestrator_graph(session_id: str, identity: Optional[Identity] = None):
    """按 session 缓存编译后的图（含 session 专属业务 agent 注入）。

    返回 None 表示该会话**不具备图链路依赖**（会话级 agent 不是真实
    `TaskClassificationAgent`，取不到 `ClassificationProcessor`）。调用方据此
    降级回 legacy 链路：生产路径的会话级 agent 恒为真实 agent，故默认走图；
    仅测试/内部注入的 legacy 形态替身（只实现 `classify_task_stream`）走 legacy。

    identity 与 legacy 一致显式透传（见 `_build_graph_deps`）；图按 `thread_id`
    缓存，仅在**会话级 agent 实例变化**时失效重建（见 `_graph_bound_agent`）。
    """
    from agents.graph import build_graph

    thread_id = _graph_thread_id(session_id)
    # 持久化模式下 loop 变化（如测试多次 asyncio.run）→ 旧编译图仍持有绑定到
    # 已关闭 loop 的 Saver，必须整体失效重建，否则 astream 会 RuntimeError。
    if _graph_loop_changed():
        _reset_graph_checkpointers()
    with _graph_lock:
        compiled = _graph_compiled.get(thread_id)
        # 编译图持有构建时那个会话级 agent 的强引用。会话 agent 被重建（测试注入
        # 替身、会话淘汰后重建）后必须失效重编 —— 否则图会一直跑旧 agent，其
        # KB/DB 连接（如 :memory:）可能已 dispose，导致与 legacy 语义不一致。
        agent = _session_agents.get(session_id)
        if compiled is not None and _graph_bound_agent.get(thread_id) is not agent:
            compiled = None
            _graph_compiled.pop(thread_id, None)
        if compiled is None:
            classifier, agents, deps = _build_graph_deps(session_id, identity=identity)
            if classifier is None:
                # 无可复用真实分类器 → 图不可用，交回 legacy（不缓存，避免掩盖能力缺失）。
                return None
            compiled = build_graph(
                classifier=classifier,  # 复用真实分类器；为 None 时内部惰性兜底
                agents=agents,
                checkpointer=_get_graph_checkpointer(),
                handoff_service=deps["handoff_service"],
                availability_handler=deps["availability_handler"],
                agent_router=deps["agent_router"],
                session_chat_history=deps["session_chat_history"],
            )
            _graph_compiled[thread_id] = compiled
            _graph_bound_agent[thread_id] = agent
        return compiled


def reload_graph() -> None:
    """运行期刷新图（配置/agent 变更时使用）。"""
    with _graph_lock:
        _graph_compiled.clear()


async def _restore_graph_flow_state(graph, cfg) -> Optional[str]:
    """读上一轮持久化的 flow_state，供本轮 new_state 显式恢复（P0-A-5）。

    InMemorySaver 同进程内有效；新线程 / 读不到时返回 None（按新会话分类）。
    """
    try:
        snapshot = await graph.aget_state(cfg)
    except Exception:
        return None
    values = getattr(snapshot, "values", None) or {}
    return values.get("flow_state")


async def _process_graph_stream(user_input: str, conversation, ident: Identity) -> AsyncGenerator[str, None]:
    """ORCHESTRATOR=graph 的旁路实现。

    与 legacy 共用同一份会话记录/锁/checkpoint 语义（record_human/record_ai、
    会话锁、checkpoint 门闩），仅把"分类+路由"换成 LangGraph 图；逐 token yield，
    保持 [THOUGHT]/[REPLY] 文本协议不变。

    P1 TODO：legacy 的 checkpoint 门闩（_restore_agent_checkpoint /
    _save_agent_checkpoint）目前由本函数复用；待 checkpointer 正式接管会话草稿后
    可移除，与报告"图路径与 legacy 的行为差异清单"一致。
    """
    from agents.graph import new_state

    session_id = ident.session_id
    _enter_session(session_id)
    try:
        evict_idle_sessions()
        async with _get_session_lock(session_id):
            conversation.record_human(ident, user_input)
            # 持久化 Saver 首次使用前显式建表（幂等；InMemorySaver 下为无操作）。
            await _ensure_graph_checkpointer_setup()
            graph = _build_orchestrator_graph(session_id, identity=ident)
            if graph is None:
                # 会话不具备图依赖（如注入的 legacy 形态替身）→ 本轮降级回 legacy。
                # 锁/human 记录语义与 legacy 一致：改用 agent.classify_task_stream。
                buf = ""
                partial = True
                agent = None
                is_checkpoint_ready = False
                try:
                    from contextlib import aclosing
                    from agents.context.types import ContextBudgetExceeded
                    agent = get_task_agent(session_id=session_id, identity=ident)
                    notice = _checkpoint_notice(agent, conversation, ident, user_input)
                    is_checkpoint_ready = True
                    if notice:
                        buf = "[REPLY]" + notice
                        yield buf
                    else:
                        async with aclosing(agent.classify_task_stream(user_input)) as response:
                            async for token in response:
                                buf += token
                                yield token
                    partial = False
                except ContextBudgetExceeded:
                    token = "[ERROR]上下文超过安全预算，请缩短本次输入后重试。"
                    buf += token
                    yield token
                finally:
                    _finalize_turn(conversation, ident, buf, partial, agent, is_checkpoint_ready)
                return
            thread_id = _graph_thread_id(session_id)
            cfg = {"configurable": {"thread_id": thread_id}}

            # 与 legacy 相同的 checkpoint 恢复门闩：首轮只展示草稿并重新询问。
            # agent 由 _build_orchestrator_graph -> _build_graph_deps 缓存进
            # _session_agents；这里只读缓存，不再二次构建（测试可只替换图）。
            agent = _session_agents.get(session_id)
            notice = (_checkpoint_notice(agent, conversation, ident, user_input)
                      if agent is not None else "")
            is_checkpoint_ready = agent is not None

            buf = ""
            partial = True
            try:
                if notice:
                    buf = "[REPLY]" + notice
                    yield buf
                else:
                    # P2-3：图挂起期间家长再发消息 → 只 parent_message 落库等客服，
                    # **不 resume、不重复 interrupt、不重跑图**（等价 legacy「已转人工短路」：
                    # 会话保持挂起态，家长原话进 transcript / 工单消息）。客服回复才走
                    # `api/handoff.py` 的 `Command(resume=...)` 恢复。
                    handoff_service = _graph_handoff_service(session_id)
                    if (handoff_service is not None
                            and await _graph_pending_interrupt(graph, cfg) is not None
                            and handoff_service.is_handed_off(session_id)):
                        handoff_service.record_input(session_id, user_input)
                        try:
                            handoff_service.parent_message(session_id, user_input)
                        except Exception as exc:  # noqa: BLE001 - 落库失败不阻塞回复
                            logger.warning("记录家长消息失败(已忽略): %s", exc)
                        copy = handoff_service.handed_off_reply()
                        buf = "[REPLY]"
                        yield buf
                        for char in copy:
                            buf += char
                            yield char
                    else:
                        # P0-A-5：每轮请求前把 flow_state 从图 state 恢复。
                        flow_state = await _restore_graph_flow_state(graph, cfg)
                        graph_input = new_state(user_input, session_id,
                                                flow_state=flow_state)
                        # 资源清理语义对齐 legacy：用 aclosing 包住 astream，
                        # 使外层 aclose() 能传播到节点内的 agent 流 → 关闭底层模型。
                        # 注：cancel（任务被取消）场景 LangGraph 不执行节点内 finally，
                        # 该差异无法在应用层修复，属已知限制（见报告）。
                        from contextlib import aclosing
                        async with aclosing(
                                graph.astream(graph_input, cfg, stream_mode="custom")
                        ) as graph_stream:
                            async for token in graph_stream:
                                if not isinstance(token, str):
                                    continue
                                buf += token
                                yield token
                partial = False
            finally:
                _finalize_turn(conversation, ident, buf, partial, agent, is_checkpoint_ready)
    finally:
        _exit_session(session_id)
        evict_idle_sessions()


async def _graph_pending_interrupt(graph, cfg) -> Optional[Any]:
    """读取会话是否停在 interrupt（未完成的任务）。"""
    try:
        snapshot = await graph.aget_state(cfg)
    except Exception:
        return None
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        if getattr(task, "interrupts", None):
            return task.interrupts[0]
    return None


# ---------------------------------------------------------------------------
# P2：转人工挂起的恢复入口（api/handoff.py 的 reply / resolve 调用）
#
# 图路径下，转人工会话停在 `interrupt()`；客服 `reply` / `resolve` 需要把图从挂起态
# 恢复。这里集中实现「查挂起 → resume → 收集输出」，并保证 **legacy 下完全 no-op**
# （ORCHESTRATOR != graph 或该 session 未走图时，reply/resolve 的既有行为逐字不变）。
# ---------------------------------------------------------------------------

def _graph_handoff_service(session_id: str):
    """取该 session 的会话级 handoff_service（图 / legacy 共用同一实例）。

    供 P2-3 判断挂起态家长输入走哪条路；取不到时返回 None（退回跑图，由其 A2 分支兜底）。
    """
    agent = _session_agents.get(session_id)
    processor = getattr(agent, "classification_processor", None)
    return getattr(processor, "handoff_service", None)


def _graph_handoff_enabled() -> bool:
    """图路径且启用 handoff 挂起：仅在 ORCHESTRATOR=graph 时成立。"""
    return _env_orchestrator() == "graph"


async def _drain_graph_resume(graph, cfg, value) -> str:
    """用 `Command(resume=value)` 恢复挂起图，返回恢复过程外发的文本。"""
    from langgraph.types import Command

    parts: list = []
    async for token in graph.astream(Command(resume=value), cfg, stream_mode="custom"):
        if isinstance(token, str):
            parts.append(token)
    return "".join(parts)


def _handoff_session_identity(session_id: str) -> Optional[Identity]:
    """按 session 归属构造请求身份（供落库客服回复 / 复用会话历史）。"""
    try:
        conversation = get_conversation_service()
        row = conversation.db_router.identity_repo.get_session(session_id)
    except Exception:
        return None
    if not row:
        return None
    from config.identity import Identity
    return Identity(user_id=row["user_id"], student_id=row.get("student_id"),
                    session_id=session_id)


def _record_graph_resume_output(session_id: str, text: str) -> None:
    """把恢复产生的可见输出落进会话历史（过滤 [THOUGHT] 后，失败静默）。"""
    cleaned = strip_thought_output(text)
    if not cleaned:
        return
    ident = _handoff_session_identity(session_id)
    if ident is None:
        return
    try:
        get_conversation_service().record_ai(ident, cleaned)
    except Exception as exc:  # noqa: BLE001 - 落库失败不影响客服回复主流程
        logger.warning("记录图恢复输出失败: %s", type(exc).__name__)


async def resume_graph_handoff(session_id: str, resume_value: Any) -> Optional[str]:
    """恢复挂起的图 thread，返回恢复输出的可见文本；非挂起 / legacy 返回 None。

    Args:
        session_id: 业务会话 id（内部拼成 `graph:<session_id>` 的 thread_id）。
        resume_value: 传给 `Command(resume=...)` 的值（客服回复正文 / resolve 哨兵）。
    """
    if not _graph_handoff_enabled() or not session_id:
        return None
    graph = _graph_compiled.get(_graph_thread_id(session_id))
    if graph is None:
        # 编译图未缓存：仅当启用持久化 checkpointer 时才值得重建（跨进程重启后凭
        # thread_id 找回挂起态）；进程内 InMemorySaver 无跨进程状态，无从恢复。
        from agents.graph import persistence_enabled

        if not persistence_enabled():
            return None
        try:
            graph = _build_orchestrator_graph(session_id)
        except Exception as exc:  # noqa: BLE001 - 构建失败视为不可恢复
            logger.warning("图重建失败(session=%s): %s", session_id, type(exc).__name__)
            return None
    cfg = {"configurable": {"thread_id": _graph_thread_id(session_id)}}
    if await _graph_pending_interrupt(graph, cfg) is None:
        return None
    try:
        output = await _drain_graph_resume(graph, cfg, resume_value)
    except Exception as exc:  # noqa: BLE001 - 恢复失败不应让客服接口 500
        logger.warning("图路径恢复失败(session=%s): %s", session_id, type(exc).__name__)
        return None
    _record_graph_resume_output(session_id, output)
    return output


def release_graph_handoff(session_id: str) -> None:
    """解除图会话级 handoff 内存短路（等价 legacy `resolve` 里的 `discard`）。

    `api/handoff.py` 的 resolve 会 new 一个新 service，其 `discard` 落在新实例上；
    而图节点用的是**会话级 agent 的** handoff_service（另一实例）。这里显式把该实例
    的 `_handed_off_sessions` 里的 session 移除，使 resolve 后会话真正恢复普通 AI 对话。
    legacy 路径下该函数是 no-op（不改变任何行为）。
    """
    if not _graph_handoff_enabled():
        return
    svc = _graph_handoff_service(session_id)
    sessions = getattr(svc, "_handed_off_sessions", None)
    if sessions is not None:
        try:
            sessions.discard(session_id)
        except Exception:  # noqa: BLE001 - 内存态清理失败不应影响接口
            pass


def _has_fresh_checkpoint_requirements(user_input: str) -> bool:
    """保守识别重新给出的具体需求；问候/旧确认不能解除恢复门闩。"""
    import re

    has_restart = re.match(
        r"\s*(?:(?:好的?|确认一下)[\s，,。]+)?(?:我要|我想|请)?"
        r"重新(?:预约|安排|排课|约课|上课|调课|改课|退课|推荐|查询)",
        user_input,
    )
    has_negated_request = re.search(
        r"(?:不要|别|不想|不需要|不打算|无需|勿|不再|不用|不必|拒绝)\s*(?:再)?(?:重新)?"
        r"(?:预约|安排|排课|约课|上课|调课|改课|退课|推荐|查询)", user_input,
    )
    if has_negated_request:
        return False
    refers_to_old_confirmation = re.search(r"确认|确定|同意|之前|原来|先前|刚才|说好的|照旧", user_input)
    has_acknowledgement = re.match(r"\s*(?:好|谢谢|ok\b|yes\b)", user_input, re.IGNORECASE)
    if not has_restart and (refers_to_old_confirmation or has_acknowledgement):
        return False
    has_request = re.search(
        r"试讲|预约|约课|排课|上课|调课|改课|退课|推荐|数学|语文|英语|物理|化学|生物|历史|地理|政治|科学",
        user_input,
    )
    has_details = re.search(
        r"今天|明天|后天|(?:下|本|这)?(?:周|星期)[一二三四五六日天]|\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}[点:：]|(?:初|高)[一二三中]|[一二三四五六七八九\d]+年级",
        user_input,
    )
    return has_request is not None and has_details is not None


def _restore_agent_checkpoint(agent, conversation, identity: Identity) -> str:
    """每个重建 Agent 仅恢复一次；首次续聊只展示草稿并重新询问，不执行业务。"""
    if getattr(agent, "_checkpoint_loaded", False):
        return ""
    from services.session_checkpoint import restore_checkpoint
    checkpoint = conversation.load_checkpoint(identity)
    if checkpoint is None:
        agent._checkpoint_loaded = True
        return ""
    result = restore_checkpoint(agent, checkpoint)
    agent._restored_drafts = result["drafts"]
    agent._requires_reconfirmation = result["requires_reconfirmation"]
    agent._checkpoint_loaded = True
    return result["notice"]


def _save_agent_checkpoint(agent, conversation, identity: Identity) -> None:
    from services.session_checkpoint import capture_checkpoint
    try:
        payload = capture_checkpoint(agent)
        if getattr(agent, "_requires_reconfirmation", False):
            payload = {**payload, "requires_reconfirmation": True,
                       "drafts": getattr(agent, "_restored_drafts", payload["drafts"])}
        conversation.save_checkpoint(identity, payload["source_flow"], payload, pending_ops=[])
    except Exception as exc:
        logger.warning("checkpoint 保存失败: %s", type(exc).__name__)


def _resolve_entry_identity(session_id: Optional[str], identity: Optional["Identity"]) -> "Identity":
    """确定本次请求身份:显式参数 > ContextVar > legacy 兜底。"""
    from config.identity import get_current_identity, legacy_identity
    ident = identity or get_current_identity()
    if ident is None:
        # 向后兼容:无身份的内部调用(demo / 测试)走 legacy(default_user),
        # 但 legacy 身份不会与匿名 web 身份的 namespace 混。
        ident = legacy_identity(session_id or _global_session_id)
    if session_id:
        ident = ident.with_session(session_id)
    elif ident.legacy and not ident.session_id:
        ident = ident.with_session(_global_session_id)
    return ident


def _enter_session(session_id: Optional[str]) -> bool:
    """标记 session 有在途请求(供缓存淘汰保护),返回是否首次进入。"""
    if not session_id:
        return False
    with _async_locks_guard:
        first = _active_sessions.get(session_id, 0) == 0
        _active_sessions[session_id] = _active_sessions.get(session_id, 0) + 1
    with _session_lock:
        _session_last_access[session_id] = time.monotonic()
    return first


def _exit_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _async_locks_guard:
        cur = _active_sessions.get(session_id, 0) - 1
        if cur <= 0:
            _active_sessions.pop(session_id, None)
        else:
            _active_sessions[session_id] = cur


# strip_thought_output 从服务层复用(单一实现,避免两处漂移)
