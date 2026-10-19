"""LangGraph P2 人工接管 `interrupt()` / `Command(resume=)` 实跑演示（离线，exit 0）。

对 `.claude/tasks/langgraph-migrate-p2.md`：图路径下用 LangGraph 原生 `interrupt()`
表达"转人工挂起"，客服 `reply` 用 `Command(resume=...)` 恢复，`resolve` 结束挂起。
演示完整链路：

    关键词触发 → 图挂起（interrupt payload）
    → 挂起期家长消息落库 parent（不 resume）
    → 客服 reply（走真实 api/handoff.py 端点）→ resume
    → 家长轮询 GET session messages 读到 parent + agent
    → resolve 结束挂起 → 会话恢复普通 AI 对话

- 用**临时 DB 文件**，不污染项目库 `data/smart_appointment.db`；
- 不连 LLM / 真实业务服务（注入替身分类器与业务 agent）；
- 全链路断言通过时 exit 0，否则 exit 1。

运行：python scripts/demo_langgraph_handoff.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.graph import HANDOFF_RESOLVED, build_graph, new_state  # noqa: E402


class FakeClassifier:
    """可指名默认意图的替身分类器。"""

    def __init__(self, default="unrelated"):
        self.default = default

    async def classify_task(self, task):
        return self.default


class FakeTrialAgent:
    def __init__(self):
        self.calls = []

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield "[REPLY][试讲]已记录您的预约意向"


class _StubTaskAgent:
    """图路径所需的最小会话级 agent 契约（不调用 LLM、不走 classify_task_stream）。

    - `classification_processor.handoff_service`：供 P2-3 挂起态判定与 reply/resolve；
    - `_checkpoint_loaded=True`：跳过 checkpoint 恢复门闩（无历史 checkpoint）；
    - 图路径 `_build_orchestrator_graph` 要求编译图绑定的 agent 与会话 agent 为同一实例，
      由 `_run` 中 `chat_handler._graph_bound_agent[THREAD_ID] = stub` 完成绑定，否则
      注入的图会被丢弃并回落到 legacy（`classify_task_stream`）。
    """

    def __init__(self, handoff_service):
        from types import SimpleNamespace

        self.classification_processor = SimpleNamespace(handoff_service=handoff_service)
        self._checkpoint_loaded = True


def _make_handoff(db_router):
    """在给定 db_router 上构造真实 HumanHandoffService（复用同一临时库）。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService.__new__(HumanHandoffService)
    svc.db_router = db_router
    svc.repo = db_router.handoff_repo
    svc.threshold = svc._resolve_threshold(None)
    svc._kickback_counts = {}
    svc._handed_off_sessions = set()
    svc._recent_inputs = {}
    return svc


SESSION_ID = "demo-handoff"
THREAD_ID = f"graph:{SESSION_ID}"
CFG = {"configurable": {"thread_id": THREAD_ID}}


async def _turn(graph, payload):
    parts = []
    async for token in graph.astream(payload, CFG, stream_mode="custom"):
        if isinstance(token, str):
            parts.append(token)
    return "".join(parts)


def _pending(snapshot):
    for task in getattr(snapshot, "tasks", None) or ():
        if getattr(task, "interrupts", None):
            return task.interrupts[0]
    return None


async def _drain(gen):
    parts = []
    async for token in gen:
        parts.append(token)
    return "".join(parts)


async def _run(tmpdir):
    from api import chat_handler
    from api.handoff import (ReplyHandoffRequest, reply_handoff,
                             resolve_handoff, ResolveHandoffRequest)
    from config.identity import legacy_identity
    from db.db_router import DatabaseRouter
    from services import human_handoff_service as hhs_module
    from services.conversation_service import ConversationService

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    db_path = "sqlite:///" + (Path(tmpdir) / "p2_demo.db").as_posix()
    os.environ["ORCHESTRATOR"] = "graph"

    router = DatabaseRouter(db_path)
    conversation = ConversationService(db_router=router)

    handoff = _make_handoff(router)

    def _patched_init(self, db_path_=None, threshold=None):
        """把 api/handoff.py 里 new 出来的 service 接到同一临时库（不触项目库）。"""
        self.db_router = router
        self.repo = router.handoff_repo
        self.threshold = self._resolve_threshold(threshold)
        self._kickback_counts = {}
        self._handed_off_sessions = set()
        self._recent_inputs = {}

    hhs_module.HumanHandoffService.__init__ = _patched_init

    trial = FakeTrialAgent()
    graph = build_graph(classifier=FakeClassifier("trial"), agents={"trial": trial},
                        checkpointer=InMemorySaver(), handoff_service=handoff)

    # 把会话级依赖接进 chat_handler（等价生产 _build_orchestrator_graph 的产物）。
    chat_handler._graph_compiled[THREAD_ID] = graph
    chat_handler._conversation_service = conversation
    stub_agent = _StubTaskAgent(handoff)
    chat_handler._session_agents[SESSION_ID] = stub_agent
    chat_handler._graph_bound_agent[THREAD_ID] = stub_agent
    ident = conversation.ensure_session(legacy_identity(SESSION_ID))

    print("=" * 78)
    print("LangGraph P2 人工接管演示：interrupt() 挂起 → Command(resume=) 恢复")
    print("=" * 78)
    print(f"临时 DB  : {db_path}")
    print(f"thread   : {THREAD_ID}")

    # 1) 关键词触发 → 挂起
    out = await _turn(graph, new_state("我要转人工", SESSION_ID))
    snap = await graph.aget_state(CFG)
    pending = _pending(snap)
    record = router.handoff_repo.get_pending()[0]
    print("\n[1] 关键词触发 → 图挂起")
    print(f"  本轮文案 : {out}")
    print(f"  图 next  : {snap.next!r}")
    print(f"  interrupt: {pending.value if pending else None}")
    print(f"  DB 记录  : id={record['id']} type={record['trigger_type']} "
          f"status={record['status']}")
    print(f"  transcript: {record['transcript']}")
    check(out.endswith("已尽快与您联系。") or "已为您转接人工顾问老师" in out, "触发文案缺失")
    check(snap.next == ("handoff_wait_node",), "应停在 handoff_wait_node")
    check(pending is not None and pending.value["trigger_type"] == "keyword",
          "interrupt payload 应为 keyword")
    check(record["transcript"] is not None, "transcript 应非空")

    # 2) 挂起期家长消息 → 落库 parent、不 resume
    parent_out = await _drain(chat_handler._process_graph_stream("大概什么时候能回复我", conversation, ident))
    snap2 = await graph.aget_state(CFG)
    parent_msgs = handoff.get_messages_by_session(SESSION_ID)
    print("\n[2] 挂起期家长再发消息 → 落库 parent，不 resume")
    print(f"  本轮文案 : {parent_out}")
    print(f"  图 next  : {snap2.next!r}（仍挂起）")
    print(f"  工单消息 : {[(m['role'], m['content']) for m in parent_msgs]}")
    check(parent_out == "[REPLY]您的会话已转人工顾问处理中，请稍候。如需加急可拨打前台电话咨询。",
          "家长消息应回已转人工短路文案")
    check([m["role"] for m in parent_msgs] == ["parent"], "家长消息应落库 role=parent")
    check(snap2.next == ("handoff_wait_node",), "不应 resume（挂起态应仍在）")

    # 3) 客服 reply（真实端点）→ Command(resume=)
    reply_resp = await reply_handoff(record["id"], ReplyHandoffRequest(
        agent_id="客服小美", content="您好，我是小美，周六上午可安排试听"))
    snap3 = await graph.aget_state(CFG)
    print("\n[3] 客服 reply（api/handoff.py 端点）→ resume")
    print(f"  reply 返回: resumed={reply_resp.data.get('resumed')} "
          f"output={reply_resp.data.get('resume_output')!r}")
    print(f"  图 next  : {snap3.next!r}（挂起已结束）")
    check(reply_resp.data.get("resumed") is True, "reply 应触发 resume")
    check("小美" in (reply_resp.data.get("resume_output") or ""), "恢复输出应含客服回复")
    check(snap3.next == (), "resume 后不应再挂起")

    # 4) 家长轮询 → 读到 parent + agent（直接调服务层；HTTP 层归属校验已移到
    #    tests/test_handoff_session_ownership.py 验证，demo 不依赖请求上下文）
    msgs = handoff.get_messages_by_session(SESSION_ID)
    print("\n[4] 家长轮询（服务层 get_messages_by_session；HTTP 归属校验见 tests/test_handoff_session_ownership.py）")
    print(f"  消息     : {[(m['role'], m['content']) for m in msgs]}")
    check([m["role"] for m in msgs] == ["parent", "agent"], "应读到 parent + agent")

    # 5) resolve → 结束挂起 → 会话恢复普通 AI 对话
    #    先把会话重新挂起（re-trigger），再 resolve。
    await _turn(graph, new_state("转人工", SESSION_ID))
    record2 = router.handoff_repo.get_pending()
    check(len(record2) == 1, "已转人工应幂等（仍 1 条 pending）")
    resolve_resp = await resolve_handoff(record["id"], ResolveHandoffRequest(handled_by="客服小美"))
    snap4 = await graph.aget_state(CFG)
    print("\n[5] resolve（api/handoff.py 端点）→ 结束挂起")
    print(f"  resolve 返回: resumed={resolve_resp.data.get('resumed')} status={resolve_resp.data.get('status')}")
    print(f"  图 next  : {snap4.next!r}  handoff={snap4.values.get('handoff')}")
    check(resolve_resp.data.get("resumed") is True, "resolve 应结束挂起")
    check(snap4.next == () and snap4.values.get("handoff") is False, "resolve 后不应挂起")

    # 6) 会话恢复普通 AI 对话
    normal = await _turn(graph, new_state("我想约一节数学试听课", SESSION_ID))
    print("\n[6] resolve 后继续正常对话")
    print(f"  本轮文案 : {normal}")
    print(f"  trial 调用: {trial.calls}")
    check("[REPLY][试讲]" in normal, "应恢复业务转发")
    check("[REPLY]您的会话已转人工顾问处理中" not in normal, "不应再走人工短路")

    router.close()

    print("\n" + "=" * 78)
    if failures:
        print("结论 : FAIL")
        for item in failures:
            print(f"  - {item}")
        print("=" * 78)
        return 1
    print("结论 : PASS（触发挂起 → 家长落库 → 客服 resume → 轮询读到 → resolve 复原）")
    print("=" * 78)
    return 0


def main():
    tmpdir = tempfile.mkdtemp(prefix="lg-p2-demo-")
    try:
        return asyncio.run(_run(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)   # Windows 下连接未释放也不报错


if __name__ == "__main__":
    raise SystemExit(main())
