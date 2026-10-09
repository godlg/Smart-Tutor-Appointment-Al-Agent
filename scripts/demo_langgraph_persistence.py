"""LangGraph P1 持久化实跑演示（离线，无外部依赖，exit 0）。

对 `.claude/tasks/langgraph-migrate-p1.md` 第二节：用 `AsyncSqliteSaver` 把图
state（含 `flow_state` / `messages`）落到 SQLite，演示「写入 → 重开连接 → 读回」，
并验证 `flow_state="trial_confirm"` 跨 Saver 实例（模拟进程重启）不丢。

- 用**临时 DB 文件**，不污染项目库 `data/smart_appointment.db`；
- 不连 LLM / 真实业务服务（注入替身分类器与业务 agent）；
- 三组证据齐全时 exit 0，否则 exit 1。

运行：python scripts/demo_langgraph_persistence.py
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from agents.graph import build_checkpointer, build_graph, new_state  # noqa: E402
from agents.task_classification.state_manager import FlowState  # noqa: E402


class FakeClassifier:
    async def classify_task(self, task):
        return "trial"


class FakeTrialAgent:
    """替身：产出试讲回复，并把共享状态推进到 TRIAL_CONFIRM（模拟真实 TrialAgent）。"""

    def __init__(self):
        self.state = type("S", (), {"value": FlowState.CLASSIFY})()
        self.calls = []

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        yield "[REPLY][试讲]已记录："
        yield user_input
        self.state.value = FlowState.TRIAL_CONFIRM


THREAD_ID = "graph:demo-persist"
CFG = {"configurable": {"thread_id": THREAD_ID}}
FIRST_TURN = "想约一节数学试听课"
SECOND_TURN = "重启后再问一句"


def _contents(snapshot):
    return [
        m.content if hasattr(m, "content") else m.get("content")
        for m in (snapshot.values or {}).get("messages", [])
    ]


async def _write_turn(db_path, text, agent):
    """新建 Saver + 编译图，写入一轮；返回读到的 flow_state / messages。"""
    saver = build_checkpointer(db_path=db_path, persistent=True)
    await saver.setup()
    graph = build_graph(classifier=FakeClassifier(), agents={"trial": agent},
                        checkpointer=saver)
    tokens = []
    async for token in graph.astream(new_state(text, THREAD_ID), CFG, stream_mode="custom"):
        if isinstance(token, str):
            tokens.append(token)
    snapshot = await graph.aget_state(CFG)
    await saver.conn.close()          # 丢弃 Saver / 编译图（模拟进程结束）
    return "".join(tokens), snapshot


async def _read_state(db_path):
    """用**新 Saver 实例**重开同一 DB，只读回 state（不写）。"""
    saver = build_checkpointer(db_path=db_path, persistent=True)
    await saver.setup()
    graph = build_graph(classifier=FakeClassifier(),
                        agents={"trial": FakeTrialAgent()}, checkpointer=saver)
    snapshot = await graph.aget_state(CFG)
    await saver.conn.close()
    return snapshot


def _tables(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def main():
    print("=" * 78)
    print("LangGraph P1 持久化演示：AsyncSqliteSaver 写入 → 重开连接 → 读回")
    print("=" * 78)

    tmpdir = tempfile.mkdtemp(prefix="lg-p1-demo-")
    db_path = os.path.join(tmpdir, "langgraph_persist.db")
    print(f"临时 DB : {db_path}")
    print(f"thread  : {THREAD_ID}")

    failures = []

    # 证据 1：写入
    agent = FakeTrialAgent()
    out1, snap1 = asyncio.run(_write_turn(db_path, FIRST_TURN, agent))
    flow1 = (snap1.values or {}).get("flow_state")
    print("\n[证据 1] 写入（新建 Saver 实例 A）")
    print(f"  第 1 轮输入 : {FIRST_TURN}")
    print(f"  流式输出    : {out1!r}")
    print(f"  flow_state  : {flow1!r}")
    print(f"  messages    : {_contents(snap1)!r}")
    if flow1 != "trial_confirm":
        failures.append(f"写入后 flow_state 应为 trial_confirm，实际 {flow1!r}")

    # 证据 2：重开连接（新 Saver 实例 B）读回
    snap2 = asyncio.run(_read_state(db_path))
    flow2 = (snap2.values or {}).get("flow_state")
    print("\n[证据 2] 重开连接（新 Saver 实例 B，模拟进程重启）")
    print(f"  flow_state  : {flow2!r}")
    print(f"  messages    : {_contents(snap2)!r}")
    if snap2 is None or flow2 is None:
        failures.append("重开后读不到 state")
    elif flow2 != flow1:
        failures.append(f"重开后 flow_state 不一致：写入 {flow1!r} vs 读回 {flow2!r}")
    if FIRST_TURN not in _contents(snap2):
        failures.append("重开后读不到第 1 轮的 messages")

    # 证据 3：同一 session 第二轮（新 Saver 实例 C）能看到第一轮 messages 且累计
    _out3, snap3 = asyncio.run(_write_turn(db_path, SECOND_TURN, FakeTrialAgent()))
    contents3 = _contents(snap3)
    print("\n[证据 3] 同一 thread 第二轮（新 Saver 实例 C，跨实例累计消息）")
    print(f"  第 2 轮输入 : {SECOND_TURN}")
    print(f"  messages    : {contents3!r}")
    print(f"  flow_state  : {(snap3.values or {}).get('flow_state')!r}")
    if FIRST_TURN not in contents3:
        failures.append("第二轮看不到第一轮的 messages（跨实例未累计）")
    if SECOND_TURN not in contents3:
        failures.append("第二轮自己的 messages 未落盘")

    # 附带：表共存核对（不改既有表）
    tables = _tables(db_path)
    print("\n[核对] checkpointer 建表（与既有业务表同库共存，不改既有表）")
    print(f"  本演示库表 : {sorted(tables)}")
    if not {"checkpoints", "writes"} <= tables:
        failures.append(f"checkpointer 表未创建：{sorted(tables)}")

    print("\n" + "=" * 78)
    if failures:
        print("结论 : FAIL")
        for item in failures:
            print(f"  - {item}")
        print("=" * 78)
        return 1
    print("结论 : PASS（写入 → 重开连接 → 读回 flow_state 三组证据齐全）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
