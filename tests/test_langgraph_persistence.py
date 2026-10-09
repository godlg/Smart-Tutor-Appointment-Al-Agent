"""LangGraph P1：SqliteSaver 持久化测试。

覆盖 `.claude/tasks/langgraph-migrate-p1.md` 验收标准 1：

1. `build_checkpointer(persistent=True)` → `AsyncSqliteSaver`；`False` → `InMemorySaver`。
2. 用 Saver 写入一个 thread 的 state（`flow_state="trial_confirm"`）→ 丢弃 Saver /
   编译图 → 用新实例重开同一 DB → 读到相同 `flow_state`（跨实例恢复）。
3. 同一 session 第二轮能看到第一轮的 `messages`（持久化版）。
4. `GRAPH_PERSISTENCE` 未设 / 为 0 时走 `InMemorySaver`（默认不变）。

无 pytest-asyncio，异步统一 `asyncio.run` 同步包装（与既有图测试一致）。
所有持久化用例均用**临时 DB 文件**，不碰项目库。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from agents.graph import (  # noqa: E402
    build_checkpointer,
    build_graph,
    new_state,
    persistence_enabled,
    resolve_db_path,
)
from agents.task_classification.state_manager import FlowState  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class FakeClassifier:
    def __init__(self, default="trial"):
        self.default = default

    async def classify_task(self, task):
        return self.default


class _SharedState:
    def __init__(self, value=None):
        self.value = value or FlowState.CLASSIFY


class FakeAgent:
    """业务替身：转发文本，运行后把共享状态推进到 `post_state`（模拟真实 agent）。"""

    def __init__(self, post_state=None):
        self.state = _SharedState()
        self.post_state = post_state
        self.calls = []

    async def run_stream(self, user_input=None, intent=None):
        self.calls.append(user_input)
        yield "[REPLY]ok"
        if self.post_state:
            self.state.value = FlowState(self.post_state)


@pytest.fixture(autouse=True)
def _clean_graph_caches():
    """清掉 chat_handler 的图 / checkpointer / Saver 缓存，避免用例间串库。"""
    from api import chat_handler

    chat_handler._reset_graph_checkpointers()
    yield
    chat_handler._reset_graph_checkpointers()


# ----------------------------------------------------------------------
# 1. build_checkpointer 类型
# ----------------------------------------------------------------------

class TestBuildCheckpointerType:

    def test_persistent_true_returns_async_sqlite_saver(self, tmp_path):
        async def _go():
            saver = build_checkpointer(db_path=str(tmp_path / "p1.db"), persistent=True)
            await saver.setup()
            return saver

        saver = _run(_go())
        assert isinstance(saver, AsyncSqliteSaver)
        _run(saver.conn.close())

    def test_persistent_false_returns_in_memory_saver(self, tmp_path):
        saver = build_checkpointer(db_path=str(tmp_path / "p1.db"), persistent=False)
        assert isinstance(saver, InMemorySaver)

    def test_persistent_requires_running_loop(self, tmp_path):
        with pytest.raises(RuntimeError):
            build_checkpointer(db_path=str(tmp_path / "p1.db"), persistent=True)


class TestResolveDbPath:

    def test_explicit_path_wins(self, tmp_path):
        target = str(tmp_path / "explicit.db")
        assert resolve_db_path(target) == target

    def test_relative_path_anchored_to_project_root(self):
        assert resolve_db_path("data/smart_appointment.db") == str(
            PROJECT_ROOT / "data" / "smart_appointment.db"
        )

    def test_project_default_db_is_existing_path(self):
        # 缺省解析到项目既有库（与 config/database.py 默认一致），不新建第三方目录。
        assert resolve_db_path() == str(PROJECT_ROOT / "data" / "smart_appointment.db")


# ----------------------------------------------------------------------
# 2. 跨实例恢复
# ----------------------------------------------------------------------

class TestCrossInstanceRecovery:

    def test_flow_state_survives_new_saver_instance(self, tmp_path):
        db = str(tmp_path / "recover.db")
        agent = FakeAgent(post_state=FlowState.TRIAL_CONFIRM.value)
        cfg = {"configurable": {"thread_id": "graph:recover"}}

        async def write():
            saver = build_checkpointer(db_path=db, persistent=True)
            await saver.setup()
            graph = build_graph(classifier=FakeClassifier("trial"),
                                agents={"trial": agent}, checkpointer=saver)
            async for _ in graph.astream(
                new_state("想约一节试听课", "graph:recover"), cfg, stream_mode="custom",
            ):
                pass
            snapshot = await graph.aget_state(cfg)
            await saver.conn.close()          # 丢弃 Saver / 编译图，模拟进程重启
            return (snapshot.values or {}).get("flow_state")

        written = _run(write())
        assert written == "trial_confirm"

        async def read():
            saver = build_checkpointer(db_path=db, persistent=True)
            await saver.setup()
            graph = build_graph(classifier=FakeClassifier("trial"),
                                agents={"trial": FakeAgent()}, checkpointer=saver)
            snapshot = await graph.aget_state(cfg)
            await saver.conn.close()
            return snapshot

        snapshot = _run(read())
        assert snapshot is not None
        assert (snapshot.values or {}).get("flow_state") == "trial_confirm"

    def test_second_turn_sees_first_turn_messages(self, tmp_path):
        db = str(tmp_path / "messages.db")
        agent = FakeAgent()
        cfg = {"configurable": {"thread_id": "graph:msg"}}

        async def two_turns():
            saver = build_checkpointer(db_path=db, persistent=True)
            await saver.setup()
            graph = build_graph(classifier=FakeClassifier("trial"),
                                agents={"trial": agent}, checkpointer=saver)
            for text in ("第一轮：我想试听", "第二轮：再问一句"):
                async for _ in graph.astream(
                    new_state(text, "graph:msg"), cfg, stream_mode="custom",
                ):
                    pass
            snapshot = await graph.aget_state(cfg)
            await saver.conn.close()
            return snapshot

        snapshot = _run(two_turns())
        contents = [
            m.content if hasattr(m, "content") else m.get("content")
            for m in snapshot.values["messages"]
        ]
        assert "第一轮：我想试听" in contents
        assert "第二轮：再问一句" in contents
        assert agent.calls == ["第一轮：我想试听", "第二轮：再问一句"]

    def test_second_turn_messages_survive_new_instance(self, tmp_path):
        db = str(tmp_path / "messages-recover.db")
        cfg = {"configurable": {"thread_id": "graph:msg2"}}

        async def write_first():
            saver = build_checkpointer(db_path=db, persistent=True)
            await saver.setup()
            graph = build_graph(classifier=FakeClassifier("trial"),
                                agents={"trial": FakeAgent()}, checkpointer=saver)
            async for _ in graph.astream(
                new_state("第一轮：我想试听", "graph:msg2"), cfg, stream_mode="custom",
            ):
                pass
            await saver.conn.close()

        _run(write_first())

        async def second_turn_new_instance():
            saver = build_checkpointer(db_path=db, persistent=True)
            await saver.setup()
            graph = build_graph(classifier=FakeClassifier("trial"),
                                agents={"trial": FakeAgent()}, checkpointer=saver)
            async for _ in graph.astream(
                new_state("第二轮：重启后继续", "graph:msg2"), cfg, stream_mode="custom",
            ):
                pass
            snapshot = await graph.aget_state(cfg)
            await saver.conn.close()
            return snapshot

        snapshot = _run(second_turn_new_instance())
        contents = [
            m.content if hasattr(m, "content") else m.get("content")
            for m in snapshot.values["messages"]
        ]
        assert "第一轮：我想试听" in contents
        assert "第二轮：重启后继续" in contents


# ----------------------------------------------------------------------
# 3. 既有业务表共存（checkpointer 另建表，不改既有表）
# ----------------------------------------------------------------------

class TestCoexistsWithExistingTables:

    def test_checkpointer_adds_tables_without_touching_existing(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "coexist.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE business_a (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO business_a (v) VALUES ('seed')")
        conn.commit()
        conn.close()

        async def _go():
            saver = build_checkpointer(db_path=str(db_path), persistent=True)
            await saver.setup()
            await saver.conn.close()

        _run(_go())

        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        rows = list(conn.execute("SELECT v FROM business_a"))
        conn.close()
        assert "business_a" in tables
        assert rows == [("seed",)]
        assert {"checkpoints", "writes"} <= tables


# ----------------------------------------------------------------------
# 4. GRAPH_PERSISTENCE 开关（默认不改）
# ----------------------------------------------------------------------

class TestPersistenceSwitch:

    def test_persistence_enabled_defaults_off(self, monkeypatch):
        monkeypatch.delenv("GRAPH_PERSISTENCE", raising=False)
        assert persistence_enabled() is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_persistence_enabled_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("GRAPH_PERSISTENCE", value)
        assert persistence_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_persistence_enabled_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("GRAPH_PERSISTENCE", value)
        assert persistence_enabled() is True

    def test_get_graph_checkpointer_default_in_memory(self, monkeypatch, tmp_path):
        from api import chat_handler

        monkeypatch.delenv("GRAPH_PERSISTENCE", raising=False)
        assert isinstance(chat_handler._get_graph_checkpointer(), InMemorySaver)

    def test_get_graph_checkpointer_zero_is_in_memory(self, monkeypatch, tmp_path):
        from api import chat_handler

        monkeypatch.setenv("GRAPH_PERSISTENCE", "0")
        assert isinstance(chat_handler._get_graph_checkpointer(), InMemorySaver)

    def test_get_graph_checkpointer_persistent_inside_loop(self, monkeypatch, tmp_path):
        from api import chat_handler

        monkeypatch.setenv("GRAPH_PERSISTENCE", "1")
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'switch.db'}")

        async def _go():
            saver = chat_handler._get_graph_checkpointer()
            await chat_handler._ensure_graph_checkpointer_setup()
            # 同一 loop 内二次获取必须复用同一实例（不每轮重连）。
            again = chat_handler._get_graph_checkpointer()
            return saver, again

        saver, again = _run(_go())
        assert isinstance(saver, AsyncSqliteSaver)
        assert saver is again
        _run(saver.conn.close())

    def test_get_graph_checkpointer_persistent_without_loop_falls_back(self, monkeypatch):
        from api import chat_handler

        monkeypatch.setenv("GRAPH_PERSISTENCE", "1")
        # 无 running loop：退回 InMemorySaver，不因构造失败崩溃。
        assert isinstance(chat_handler._get_graph_checkpointer(), InMemorySaver)
