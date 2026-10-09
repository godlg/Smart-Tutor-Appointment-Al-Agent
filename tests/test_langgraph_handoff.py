"""LangGraph P2：`interrupt()` 原生人工接管测试。

覆盖 `.claude/tasks/langgraph-migrate-p2.md` 的验收标准：

- P2-1 关键词触发 → 图挂起（`next` 非空 / 存在 interrupt），`human_handoffs` 落 keyword 记录，transcript 非空
- P2-1 踢回阈值触发 → 同上（`trigger_type="kickback_threshold"`）
- P2-2 客服 `reply`（`Command(resume=...)`）→ 产出包含客服回复内容
- P2-3 挂起期间家长新消息 → 落库 `role="parent"` 且**未 resume**（挂起态仍存在）
- P2-4 `resolve` 之后 → 会话可继续正常对话（不再挂起）
- P2-5 家长轮询 `get_messages_by_session` 读到 parent + agent 全部消息
- legacy 不变：ORCHESTRATOR != graph 时 reply / resolve 不触碰图

既有 handoff 测试（`test_human_handoff*.py`）跑 legacy 路径，未做任何修改。

约束（与仓库其它测试一致）：无 pytest-asyncio，异步用例统一 `_run(_collect(...))`
同步包装；替身 agent / 真实 in-memory handoff_service，不触网。
"""

import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.graph import (  # noqa: E402
    HANDOFF_RESOLVED,
    build_graph,
    new_state,
    route_after_unrelated,
)
from agents.graph.build import route_entry  # noqa: E402
from config.identity import legacy_identity  # noqa: E402
from services.human_handoff_service import (  # noqa: E402
    HANDED_OFF_REPLY,
    HANDOFF_REPLY,
)


def _run(coro):
    return asyncio.run(coro)


def _cfg(session_id):
    return {"configurable": {"thread_id": f"graph:{session_id}"}}


async def _collect_async(graph, payload, cfg):
    out = []
    async for token in graph.astream(payload, cfg, stream_mode="custom"):
        if isinstance(token, str):
            out.append(token)
    return "".join(out)


def _collect(graph, payload, cfg):
    return _run(_collect_async(graph, payload, cfg))


def _pending_interrupt(snapshot):
    for task in getattr(snapshot, "tasks", None) or ():
        if getattr(task, "interrupts", None):
            return task.interrupts[0]
    return None


def _contents(snapshot):
    """图 messages 里各条文本（add_messages 会把 dict 归约成 langchain message）。"""
    out = []
    for m in (snapshot.values or {}).get("messages", []):
        out.append(m.content if hasattr(m, "content") else m.get("content"))
    return out


# 分类节点外发的可观测性标注（图路径独有，非本阶段验收关注点）。
_LANGGRAPH_THOUGHT_RE = re.compile(r"\[THOUGHT\]\[LangGraph\] 意图识别为 [^。]*。")


def _clean(text):
    return _LANGGRAPH_THOUGHT_RE.sub("", text)


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------

class FakeClassifier:
    def __init__(self, default="unrelated"):
        self.default = default
        self.calls = []

    async def classify_task(self, task):
        self.calls.append(task)
        return self.default


class FakeAgent:
    def __init__(self, name, tokens=None):
        self.name = name
        self.calls = []
        self.tokens = tokens or [f"[REPLY][{name}]已收到：{name}"]

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        for token in self.tokens:
            yield token


class FakeRouter:
    """兜底能力列表替身（未达阈值时使用）。"""

    def __init__(self):
        self.calls = 0

    async def handle_unsupported_task(self, category):
        self.calls += 1
        yield "[REPLY][归类机器人]这个问题暂时超出我的能力范围。"


@pytest.fixture
def db_router():
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


@pytest.fixture
def handoff_service(db_router):
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService.__new__(HumanHandoffService)
    svc.db_router = db_router
    svc.repo = db_router.handoff_repo
    svc.threshold = svc._resolve_threshold(None)
    svc._kickback_counts = {}
    svc._handed_off_sessions = set()
    svc._recent_inputs = {}
    return svc


def _build(classifier=None, handoff=None, agents=None, router=None, history=None):
    return build_graph(
        classifier=classifier or FakeClassifier("unrelated"),
        agents=agents if agents is not None else {"trial": FakeAgent("trial")},
        checkpointer=InMemorySaver(),
        handoff_service=handoff,
        agent_router=router,
        session_chat_history=history,
    )


# ----------------------------------------------------------------------
# P2-1：触发 → interrupt() 挂起
# ----------------------------------------------------------------------

class TestKeywordSuspend:

    def test_keyword_triggers_suspend_and_persists(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("kw")
        out = _collect(graph, new_state("我要转人工", "kw"), cfg)

        # legacy 同款文案逐字不变（本轮前端看到的仍是这段）
        assert out == "[REPLY][HANDOFF]" + HANDOFF_REPLY

        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ("handoff_wait_node",)

        pending = _pending_interrupt(snapshot)
        assert pending is not None
        assert pending.value["trigger_type"] == "keyword"
        assert pending.value["handoff_id"] is not None
        assert pending.value["session_id"] == "kw"
        assert pending.value["reason"]

        # DB 落库 + transcript 非空
        records = db_router.handoff_repo.get_pending()
        assert len(records) == 1
        assert records[0]["trigger_type"] == "keyword"
        assert records[0]["transcript"] is not None
        assert "转人工" in records[0]["transcript"]

    def test_interrupt_payload_matches_db_record(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("kw2")
        _collect(graph, new_state("转人工", "kw2"), cfg)

        record = db_router.handoff_repo.get_pending()[0]
        pending = _pending_interrupt(_run(graph.aget_state(cfg)))
        assert pending.value["handoff_id"] == record["id"]
        assert pending.value["trigger_type"] == record["trigger_type"]


class TestKickbackSuspend:

    def test_threshold_triggers_suspend_and_persists(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("kb")

        first = _collect(graph, new_state("今天天气怎么样", "kb"), cfg)
        second = _collect(graph, new_state("随便聊聊", "kb"), cfg)
        assert "[HANDOFF]" not in first + second
        assert db_router.handoff_repo.get_pending() == []

        third = _collect(graph, new_state("再问个无关的", "kb"), cfg)
        assert _clean(third) == "[REPLY][HANDOFF]" + HANDOFF_REPLY

        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ("handoff_wait_node",)
        pending = _pending_interrupt(snapshot)
        assert pending.value["trigger_type"] == "kickback_threshold"

        record = db_router.handoff_repo.get_pending()[0]
        assert record["trigger_type"] == "kickback_threshold"
        assert record["unrelated_count"] >= 3
        assert record["transcript"] is not None

    def test_below_threshold_does_not_suspend(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("kb-low")
        _collect(graph, new_state("天气一", "kb-low"), cfg)
        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ()
        assert _pending_interrupt(snapshot) is None
        assert db_router.handoff_repo.get_pending() == []


# ----------------------------------------------------------------------
# P2-2：客服 reply → Command(resume=...) 恢复
# ----------------------------------------------------------------------

class TestReplyResume:

    def test_resume_emits_customer_reply(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("rp")
        _collect(graph, new_state("转人工", "rp"), cfg)
        assert _run(graph.aget_state(cfg)).next == ("handoff_wait_node",)

        resumed = _collect(graph, Command(resume="您好，我是客服小美"), cfg)
        assert resumed == "[REPLY]您好，我是客服小美"

        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ()                      # 挂起已结束
        assert any("您好，我是客服小美" in c for c in _contents(snapshot))

    def test_resume_keeps_original_copy_once(self, handoff_service, db_router):
        """恢复只重跑挂起节点，不会重复外发/重复落库触发轮文案（幂等）。"""
        graph = _build(handoff=handoff_service)
        cfg = _cfg("rp-idem")
        _collect(graph, new_state("转人工", "rp-idem"), cfg)
        reused = _collect(graph, Command(resume="客服回复"), cfg)
        assert reused.count(HANDOFF_REPLY) == 0         # 触发文案不再重复
        contents = _contents(_run(graph.aget_state(cfg)))
        assert sum(1 for c in contents if HANDOFF_REPLY in c) == 1

    def test_agent_reply_persisted_and_pollable(self, handoff_service, db_router):
        graph = _build(handoff=handoff_service)
        cfg = _cfg("rp-db")
        _collect(graph, new_state("转人工", "rp-db"), cfg)
        record = db_router.handoff_repo.get_pending()[0]

        handoff_service.parent_message("rp-db", "大概什么时候能回复我")
        handoff_service.agent_reply(record["id"], "客服小美", "已为您预留周六名额")
        _collect(graph, Command(resume="已为您预留周六名额"), cfg)

        msgs = handoff_service.get_messages_by_session("rp-db")
        assert [m["role"] for m in msgs] == ["parent", "agent"]
        assert msgs[0]["content"] == "大概什么时候能回复我"
        assert msgs[1]["content"] == "已为您预留周六名额"


# ----------------------------------------------------------------------
# P2-4：resolve 之后 → 恢复普通 AI 对话
# ----------------------------------------------------------------------

class TestResolveEndsSuspension:

    def test_resolve_sentinel_ends_suspension(self, handoff_service, db_router):
        agents = {"trial": FakeAgent("trial")}
        graph = _build(handoff=handoff_service, agents=agents)
        cfg = _cfg("rs")

        _collect(graph, new_state("转人工", "rs"), cfg)
        record = db_router.handoff_repo.get_pending()[0]
        assert _run(graph.aget_state(cfg)).next == ("handoff_wait_node",)

        handoff_service.resolve(record["id"], handled_by="客服A")
        out = _collect(graph, Command(resume=HANDOFF_RESOLVED), cfg)
        assert out == ""                                # 哨兵不产生客服回复文案

        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ()                      # 不再挂起
        assert snapshot.values["handoff"] is False

    def test_conversation_continues_after_resolve(self, handoff_service, db_router):
        agents = {"trial": FakeAgent("trial")}
        graph = _build(classifier=FakeClassifier("trial"), handoff=handoff_service,
                       agents=agents)
        cfg = _cfg("rs-cont")

        _collect(graph, new_state("转人工", "rs-cont"), cfg)
        record = db_router.handoff_repo.get_pending()[0]
        handoff_service.resolve(record["id"], handled_by="客服A")
        _collect(graph, Command(resume=HANDOFF_RESOLVED), cfg)

        # resolve 后再发业务请求 → 正常路由到 trial agent（不再被人工接管短路）
        out = _collect(graph, new_state("想约一节试听课", "rs-cont"), cfg)
        assert "[REPLY][trial]" in out
        assert agents["trial"].calls == ["想约一节试听课"]
        assert HANDED_OFF_REPLY not in out


# ----------------------------------------------------------------------
# P2-5：家长轮询读到全部消息
# ----------------------------------------------------------------------

class TestParentPolling:

    def test_get_messages_by_session_reads_parent_and_agent(self, handoff_service):
        record = handoff_service.trigger_handoff("poll-1", "keyword", reason="人工")
        handoff_service.parent_message("poll-1", "在吗")
        handoff_service.parent_message("poll-1", "有人处理吗")
        handoff_service.agent_reply(record["id"], "客服小美", "在的，马上帮您看")

        msgs = handoff_service.get_messages_by_session("poll-1")
        assert [m["role"] for m in msgs] == ["parent", "parent", "agent"]
        assert msgs[-1]["role"] == "agent"
        assert msgs[-1]["content"] == "在的，马上帮您看"


# ----------------------------------------------------------------------
# 纯函数：转人工路由
# ----------------------------------------------------------------------

class TestHandoffRouting:

    def test_route_entry_handoff_goes_to_handoff_node(self):
        assert route_entry({"handoff": True}) == "handoff_node"

    def test_route_after_unrelated_suspend_vs_end(self):
        assert route_after_unrelated({"handoff": True}) == "handoff_wait_node"
        from langgraph.graph import END

        assert route_after_unrelated({"handoff": False}) == END


# ----------------------------------------------------------------------
# P2-3：挂起期间家长输入（api 层）→ 落库 parent、不 resume
# ----------------------------------------------------------------------

class _FakeConversation:
    def __init__(self, db_router):
        self.db_router = db_router
        self.human = []
        self.ai = []

    def record_human(self, ident, text):
        self.human.append(text)

    def record_ai(self, ident, text, partial=False):
        self.ai.append(text)

    def load_checkpoint(self, ident):
        return None

    def save_checkpoint(self, *args, **kwargs):
        pass


class _StubAgent:
    """仅用于把会话级 handoff_service 暴露给 chat_handler._graph_handoff_service。"""

    def __init__(self, handoff_service):
        self.classification_processor = SimpleNamespace(handoff_service=handoff_service)
        self._checkpoint_loaded = True


@pytest.fixture
def graph_mode(monkeypatch):
    """启用 ORCHESTRATOR=graph，并清空图 / 会话缓存，退出时复位。"""
    from api import chat_handler

    monkeypatch.setenv("ORCHESTRATOR", "graph")
    chat_handler._reset_graph_checkpointers()
    saved_agents = dict(chat_handler._session_agents)
    chat_handler._session_agents.clear()
    yield chat_handler
    chat_handler._session_agents.clear()
    chat_handler._session_agents.update(saved_agents)
    chat_handler._graph_compiled.clear()
    chat_handler._reset_graph_checkpointers()


def _drain_stream(gen):
    async def _inner():
        out = []
        async for token in gen:
            out.append(token)
        return "".join(out)

    return _run(_inner())


class TestParentInputWhileSuspended:

    def test_parent_message_logged_without_resuming(self, graph_mode, handoff_service,
                                                    db_router, monkeypatch):
        ch = graph_mode
        graph = _build(handoff=handoff_service)
        session_id = "susp-parent"
        cfg = _cfg(session_id)
        _collect(graph, new_state("我要转人工", session_id), cfg)
        assert _run(graph.aget_state(cfg)).next == ("handoff_wait_node",)

        monkeypatch.setattr(ch, "_build_orchestrator_graph", lambda sid, identity=None: graph)
        conversation = _FakeConversation(db_router)
        ch._session_agents[session_id] = _StubAgent(handoff_service)
        ident = legacy_identity(session_id)

        out = _drain_stream(ch._process_graph_stream("在吗", conversation, ident))

        # 与 legacy 已转人工短路逐字一致
        assert out == "[REPLY]" + HANDED_OFF_REPLY
        # 家长消息落库为 role="parent"
        msgs = handoff_service.get_messages_by_session(session_id)
        assert [(m["role"], m["content"]) for m in msgs] == [("parent", "在吗")]
        # 未 resume：挂起态仍在，且不产生重复触发记录
        snapshot = _run(graph.aget_state(cfg))
        assert snapshot.next == ("handoff_wait_node",)
        assert len(db_router.handoff_repo.get_pending()) == 1

    def test_after_customer_reply_parent_sees_it(self, graph_mode, handoff_service,
                                                 db_router, monkeypatch):
        ch = graph_mode
        graph = _build(handoff=handoff_service)
        session_id = "susp-reply"
        cfg = _cfg(session_id)
        _collect(graph, new_state("转人工", session_id), cfg)
        record = db_router.handoff_repo.get_pending()[0]

        monkeypatch.setattr(ch, "_build_orchestrator_graph", lambda sid, identity=None: graph)
        conversation = _FakeConversation(db_router)
        ch._session_agents[session_id] = _StubAgent(handoff_service)

        # 家长挂起期内再发消息
        _drain_stream(ch._process_graph_stream("有人吗", conversation, legacy_identity(session_id)))
        # 客服回复 → resume
        handoff_service.agent_reply(record["id"], "客服小美", "在的")
        _collect(graph, Command(resume="在的"), cfg)

        msgs = handoff_service.get_messages_by_session(session_id)
        assert [m["role"] for m in msgs] == ["parent", "agent"]
        assert msgs[1]["content"] == "在的"


# ----------------------------------------------------------------------
# P2-2：chat_handler.resume_graph_handoff 的 legacy 不变性
# ----------------------------------------------------------------------

class TestResumeHelperLegacyInvariance:

    def test_returns_none_when_orchestrator_legacy(self, monkeypatch, handoff_service):
        from api import chat_handler

        monkeypatch.setenv("ORCHESTRATOR", "legacy")
        chat_handler._reset_graph_checkpointers()
        assert _run(chat_handler.resume_graph_handoff("any", "hi")) is None

    def test_returns_none_when_not_suspended(self, monkeypatch):
        from api import chat_handler

        monkeypatch.setenv("ORCHESTRATOR", "graph")
        chat_handler._reset_graph_checkpointers()
        # 无编译图 → 视为未挂起
        assert _run(chat_handler.resume_graph_handoff("nobody", "hi")) is None


# ----------------------------------------------------------------------
# P2-4：跨实例 resolve（客服 API new 的服务 vs 图会话级服务）
# ----------------------------------------------------------------------

class TestResolveReleasesSessionShortCircuit:

    def test_release_graph_handoff_clears_session_and_allows_business(self, graph_mode,
                                                                     handoff_service,
                                                                     db_router,
                                                                     monkeypatch):
        """resolve 解除的是会话级 agent 的 handoff 内存短路（非 API new 的实例）。

        复现生产路径：图节点用会话级 handoff_service，而 api/handoff.py 的 resolve
        会 new 另一个实例；若不显式释放会话级实例的 `_handed_off_sessions`，
        resolve 后家长消息仍会被「已转人工」短路。
        """
        ch = graph_mode
        agents = {"trial": FakeAgent("trial")}
        graph = _build(classifier=FakeClassifier("trial"), handoff=handoff_service,
                       agents=agents)
        session_id = "resolve-cross"
        cfg = _cfg(session_id)
        _collect(graph, new_state("转人工", session_id), cfg)
        record = db_router.handoff_repo.get_pending()[0]
        monkeypatch.setattr(ch, "_build_orchestrator_graph", lambda sid, identity=None: graph)
        ch._session_agents[session_id] = _StubAgent(handoff_service)
        conversation = _FakeConversation(db_router)

        # 模拟客服 resolve：另一个 service 实例（与图会话级实例不同）落 DB resolved。
        from services.human_handoff_service import HumanHandoffService

        api_svc = HumanHandoffService.__new__(HumanHandoffService)
        api_svc.db_router = db_router
        api_svc.repo = db_router.handoff_repo
        api_svc.threshold = 3
        api_svc._kickback_counts = {}
        api_svc._handed_off_sessions = set()
        api_svc._recent_inputs = {}
        api_svc.resolve(record["id"], handled_by="客服A")
        assert db_router.handoff_repo.get_pending_by_session(session_id) is None

        # 显式释放会话级实例的内存短路（否则 is_handed_off 仍命中它的 set）。
        ch.release_graph_handoff(session_id)
        assert handoff_service.is_handed_off(session_id) is False

        _collect(graph, Command(resume=HANDOFF_RESOLVED), cfg)
        out = _drain_stream(ch._process_graph_stream("我想约试听课", conversation,
                                                     legacy_identity(session_id)))
        assert "[REPLY][trial]" in out
        assert agents["trial"].calls == ["我想约试听课"]

    def test_release_is_noop_in_legacy(self, monkeypatch):
        from api import chat_handler

        monkeypatch.setenv("ORCHESTRATOR", "legacy")
        chat_handler._session_agents.clear()
        # 不应抛异常，也不产生任何副作用
        assert chat_handler.release_graph_handoff("whatever") is None
