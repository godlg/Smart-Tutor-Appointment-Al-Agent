"""LangGraph PoC 测试。

覆盖（对应 DoD 第 3 条）：
1. 图编译成功
2. route_by_intent 纯函数：4 类 intent + 兜底（参数化）
3. 4 条路径各跑一次，断言落到预期业务节点（替身 agent 记录被调用）
4. checkpointer：同一 thread_id 跑两轮，第二轮能读到第一轮写入的 messages
5. interrupt() + Command(resume=...) 跑通一次，断言恢复后的输出

注：本仓库未装 pytest-asyncio，异步用例统一用同步包装 _run(coro)。
"""

import asyncio

import pytest

from agents.graph import build_graph, new_state, route_by_intent
from agents.graph.build import TaskClassifierAdapter
from agents.graph.state import TutorState


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# 替身：可注入分类器 + 业务 agent（不初始化任何服务）
# ----------------------------------------------------------------------

class FakeClassifier:
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


class FakeAgent:
    """只做转发同形态的替身：记录调用并按 token 产出文本。"""

    def __init__(self, name, tokens=None):
        self.name = name
        self.calls = []
        self.tokens = tokens or [f"[REPLY][{name}]处理完成"]

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        for token in self.tokens:
            yield token


def _make_graph(intent_mapping=None, default="unrelated"):
    classifier = FakeClassifier(intent_mapping, default=default)
    agents = {
        "consultation": FakeAgent("consultation"),
        "trial": FakeAgent("trial"),
        "lesson": FakeAgent("lesson"),
        "reschedule": FakeAgent("reschedule"),
    }
    graph = build_graph(classifier=classifier, agents=agents)
    return graph, classifier, agents


def _collect_custom(graph, payload, config):
    async def _inner():
        out = []
        async for token in graph.astream(payload, config, stream_mode="custom"):
            out.append(token)
        return out

    return _run(_inner())


# ----------------------------------------------------------------------
# 1. 图编译成功
# ----------------------------------------------------------------------

class TestGraphBuild:

    def test_build_graph_compiles(self):
        graph, _, _ = _make_graph()
        assert graph is not None
        assert hasattr(graph, "astream")

    def test_default_components_construct_without_services(self):
        """默认分类器是惰性的；不触发 LLM / KB / DB 构造即可编译。"""
        adapter = TaskClassifierAdapter()
        assert adapter._classifier is None
        graph = build_graph()
        assert graph is not None


# ----------------------------------------------------------------------
# 2. route_by_intent 纯函数
# ----------------------------------------------------------------------

class TestRouteByIntent:

    @pytest.mark.parametrize("intent,expected", [
        ("consultation", "consultation_node"),
        ("trial", "trial_node"),
        ("lesson", "lesson_node"),
        ("reschedule", "reschedule_node"),
        ("cancel", "unrelated_node"),
        ("package", "unrelated_node"),
        ("progress", "unrelated_node"),
        ("recommendation", "unrelated_node"),
        ("recommendation_confirm", "unrelated_node"),
        ("unrelated", "unrelated_node"),
        ("", "unrelated_node"),
    ])
    def test_route_maps_intent_to_node(self, intent, expected):
        assert route_by_intent({"intent": intent}) == expected

    def test_route_is_pure_no_side_effects(self):
        """纯函数：同一输入多次调用结果稳定，且不依赖 state 其它字段。"""
        state = {**new_state("hi", "s1"), "intent": "trial"}
        state["messages"] = []
        assert route_by_intent(state) == route_by_intent(state) == "trial_node"


# ----------------------------------------------------------------------
# 3. 4 条路径各跑一次，断言落到预期业务节点
# ----------------------------------------------------------------------

class TestRoutesCallExpectedAgent:

    @pytest.mark.parametrize("user_input,intent,node", [
        ("孩子数学怎么学", "consultation", "consultation"),
        ("想约一节试听课", "trial", "trial"),
        ("帮我预约明天下午3点的课", "lesson", "lesson"),
        ("把周三的课调到周五", "reschedule", "reschedule"),
    ])
    def test_route_reaches_expected_agent(self, user_input, intent, node):
        graph, classifier, agents = _make_graph(
            intent_mapping={user_input: intent}, default=intent,
        )
        cfg = {"configurable": {"thread_id": f"t-{node}"}}
        tokens = _collect_custom(graph, new_state(user_input, cfg["configurable"]["thread_id"]), cfg)

        assert agents[node].calls == [user_input]
        for other in agents:
            if other != node:
                assert agents[other].calls == []
        assert f"[REPLY][{node}]" in "".join(tokens)

    def test_unrelated_intent_falls_back(self):
        graph, classifier, agents = _make_graph(default="unrelated")
        cfg = {"configurable": {"thread_id": "t-un"}}
        tokens = _collect_custom(graph, new_state("今天天气如何", "t-un"), cfg)
        assert all(a.calls == [] for a in agents.values())
        assert "超出我的能力范围" in "".join(tokens)


# ----------------------------------------------------------------------
# 4. checkpointer：同一 thread_id 跨轮持久
# ----------------------------------------------------------------------

class TestCheckpointerPersistence:

    def test_same_thread_second_turn_reads_first_turn_messages(self):
        graph, _, agents = _make_graph(default="trial")
        cfg = {"configurable": {"thread_id": "persist-1"}}

        _collect_custom(graph, new_state("第一轮：我想试听", "persist-1"), cfg)
        _collect_custom(graph, new_state("第二轮：再问一句", "persist-1"), cfg)

        async def _state():
            return await graph.aget_state(cfg)

        snapshot = _run(_state())
        contents = [
            m.content if hasattr(m, "content") else m.get("content")
            for m in snapshot.values["messages"]
        ]
        assert "第一轮：我想试听" in contents
        assert "第二轮：再问一句" in contents
        # 第二轮确实在同一线程内累计，而非新建
        assert len(agents["trial"].calls) == 2

    def test_different_threads_are_isolated(self):
        graph, _, _ = _make_graph(default="trial")
        cfg_a = {"configurable": {"thread_id": "iso-a"}}
        cfg_b = {"configurable": {"thread_id": "iso-b"}}

        _collect_custom(graph, new_state("A 的消息", "iso-a"), cfg_a)
        _collect_custom(graph, new_state("B 的消息", "iso-b"), cfg_b)

        async def _state(cfg):
            return await graph.aget_state(cfg)

        a_contents = [m.content for m in _run(_state(cfg_a)).values["messages"]]
        b_contents = [m.content for m in _run(_state(cfg_b)).values["messages"]]
        assert any("A 的消息" == c for c in a_contents)
        assert all("B 的消息" != c for c in a_contents)
        assert any("B 的消息" == c for c in b_contents)


# ----------------------------------------------------------------------
# 5. interrupt() + Command(resume=...)
# ----------------------------------------------------------------------

class TestInterruptHandoff:

    def test_interrupt_then_resume_completes_to_end(self):
        from langgraph.types import Command

        graph, _, agents = _make_graph(default="trial")
        cfg = {"configurable": {"thread_id": "int-1"}}

        # 第一轮：含『转人工』→ trial_node interrupt，业务 agent 不应被调用
        first = _collect_custom(graph, new_state("我要转人工", "int-1"), cfg)
        assert agents["trial"].calls == []
        assert "[THOUGHT]" in "".join(first)      # 分类节点的思考已外发
        assert "[REPLY][trial]" not in "".join(first)  # 但未进入正常转发

        async def _pending():
            snap = await graph.aget_state(cfg)
            for task in snap.tasks or ():
                if task.interrupts:
                    return task.interrupts[0]
            return None

        pending = _run(_pending())
        assert pending is not None
        assert pending.value["reason"] == "human_handoff"

        # 第二轮：Command(resume=...) 恢复 → 继续到 END
        resumed = _collect_custom(graph, Command(resume="客服已介入"), cfg)
        joined = "".join(resumed)
        assert "客服已介入" in joined
        assert "人工接管" in joined

        async def _state():
            return await graph.aget_state(cfg)

        snapshot = _run(_state())
        assert snapshot.values["handoff"] is True

    def test_normal_trial_does_not_interrupt(self):
        graph, _, agents = _make_graph(default="trial")
        cfg = {"configurable": {"thread_id": "int-2"}}
        tokens = _collect_custom(graph, new_state("想约试听课", "int-2"), cfg)
        assert agents["trial"].calls == ["想约试听课"]
        assert "[REPLY][trial]" in "".join(tokens)


# ----------------------------------------------------------------------
# 6. 入口开关：默认 legacy，ORCHESTRATOR=graph 才走图
# ----------------------------------------------------------------------

class _FakeConversation:
    def __init__(self):
        self.human = []
        self.ai = []

    def ensure_session(self, ident):
        return ident

    def record_human(self, ident, text):
        self.human.append(text)

    def record_ai(self, ident, text, partial=False):
        self.ai.append(text)

    def load_checkpoint(self, ident):
        return None

    def save_checkpoint(self, *args, **kwargs):
        pass


class _FakeLegacyAgent:
    """legacy 链路替身：标记 classify_task_stream 被调用。"""

    def __init__(self):
        self.calls = []

    async def classify_task_stream(self, task):
        self.calls.append(task)
        yield "[REPLY]legacy"


def _collect_stream(gen):
    async def _inner():
        out = []
        async for token in gen:
            out.append(token)
        return out

    return _run(_inner())


class TestOrchestratorSwitch:

    def test_default_env_is_legacy(self, monkeypatch):
        from api import chat_handler
        monkeypatch.delenv("ORCHESTRATOR", raising=False)
        assert chat_handler._env_orchestrator() == "legacy"

    def test_default_goes_legacy_not_graph(self, monkeypatch):
        from api import chat_handler
        monkeypatch.delenv("ORCHESTRATOR", raising=False)
        conv = _FakeConversation()
        legacy_agent = _FakeLegacyAgent()
        monkeypatch.setattr(chat_handler, "get_conversation_service", lambda: conv)
        monkeypatch.setattr(chat_handler, "get_task_agent",
                            lambda session_id=None, identity=None: legacy_agent)
        monkeypatch.setattr(chat_handler, "_process_graph_stream",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应走图")))

        out = _collect_stream(chat_handler.ProcessUserInput_stream("你好", session_id="legacy-1"))
        assert "[REPLY]legacy" in "".join(out)
        assert legacy_agent.calls == ["你好"]
        assert conv.ai == ["[REPLY]legacy"]

    def test_graph_env_uses_graph_stream(self, monkeypatch):
        from api import chat_handler
        from config.identity import legacy_identity

        monkeypatch.setenv("ORCHESTRATOR", "graph")
        conv = _FakeConversation()
        monkeypatch.setattr(chat_handler, "get_conversation_service", lambda: conv)

        def _fake_build(session_id, identity=None):
            classifier = FakeClassifier({}, default="trial")
            agents = {"trial": FakeAgent("trial")}
            return build_graph(classifier=classifier, agents=agents)

        monkeypatch.setattr(chat_handler, "_build_orchestrator_graph", _fake_build)

        # P0-A 起图与 legacy 一致：匿名(非 legacy)身份访问业务类别会先走 B7 校验。
        # 本用例只验证"ORCHESTRATOR=graph 时确实走图路径"，故用 legacy 身份放行业务转发。
        out = _collect_stream(chat_handler.ProcessUserInput_stream(
            "想约试听课", session_id="graph-1", identity=legacy_identity("graph-1")))
        joined = "".join(out)
        assert "[REPLY][trial]" in joined
        assert conv.human == ["想约试听课"]
        assert any("trial" in a for a in conv.ai)
