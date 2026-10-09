"""LangGraph PoC 实跑演示（离线，无外部依赖，exit 0）。

打印三组证据：
1. 4 条路由各走对节点（consultation / trial / lesson / reschedule + 兜底）
2. 跨轮状态恢复（同一 thread_id 第二轮读到第一轮写入的 messages）
3. interrupt() 人工接管 + Command(resume=...) 恢复

运行：python scripts/demo_langgraph_poc.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph import build_graph, new_state  # noqa: E402


class DemoClassifier:
    def __init__(self, mapping, default="unrelated"):
        self.mapping = mapping
        self.default = default

    async def classify_task(self, task):
        for key, intent in self.mapping.items():
            if key in task:
                return intent
        return self.default


class DemoAgent:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def run_stream(self, user_input=None):
        self.calls.append(user_input)
        yield f"[THOUGHT][{self.name}] 收到请求：{user_input}"
        yield f"[REPLY][{self.name}] 已处理"


def _agents():
    return {
        "consultation": DemoAgent("咨询"),
        "trial": DemoAgent("试讲"),
        "lesson": DemoAgent("正式课"),
        "reschedule": DemoAgent("调课"),
    }


async def _collect(graph, payload, cfg):
    out = []
    async for token in graph.astream(payload, cfg, stream_mode="custom"):
        out.append(token)
    return out


async def evidence_routing():
    print("=" * 68)
    print("证据 1：4 条路由各走对节点")
    print("=" * 68)
    routes = [
        ("孩子数学怎么学", "consultation", "consultation"),
        ("想约一节试听课", "trial", "trial"),
        ("帮我预约明天的课", "lesson", "lesson"),
        ("把周三的课调到周五", "reschedule", "reschedule"),
    ]
    classifier = DemoClassifier({k: i for k, i, _ in routes}, default="unrelated")
    agents = _agents()
    graph = build_graph(classifier=classifier, agents=agents)
    ok = True
    for text, intent, expect_node in routes:
        cfg = {"configurable": {"thread_id": f"route-{expect_node}"}}
        tokens = await _collect(graph, new_state(text, cfg["configurable"]["thread_id"]), cfg)
        called = agents[expect_node].calls
        hit = called and called[-1] == text
        ok = ok and bool(hit)
        print(f"  [{ 'OK' if hit else 'FAIL' }] {text:<14} -> intent={intent:<13} "
              f"命中 agent={expect_node}  产出={''.join(tokens)[:40]}...")
    # 兜底
    cfg = {"configurable": {"thread_id": "route-unrelated"}}
    tokens = await _collect(graph, new_state("今天天气如何", "route-unrelated"), cfg)
    fallback_hit = "超出我的能力范围" in "".join(tokens)
    ok = ok and fallback_hit
    print(f"  [{ 'OK' if fallback_hit else 'FAIL' }] {'今天天气如何':<14} -> intent=unrelated      "
          f"命中 agent=unrelated_node")
    return ok


async def evidence_persistence():
    print()
    print("=" * 68)
    print("证据 2：跨轮状态恢复（同一 thread_id，checkpointer=InMemorySaver）")
    print("=" * 68)
    classifier = DemoClassifier({}, default="trial")
    agents = _agents()
    graph = build_graph(classifier=classifier, agents=agents)
    cfg = {"configurable": {"thread_id": "demo-thread"}}

    await _collect(graph, new_state("第一轮：我想试听", "demo-thread"), cfg)
    first_len = len((await graph.aget_state(cfg)).values["messages"])

    await _collect(graph, new_state("第二轮：再补一句", "demo-thread"), cfg)
    snapshot = await graph.aget_state(cfg)
    contents = [getattr(m, "content", None) for m in snapshot.values["messages"]]

    recovered = "第一轮：我想试听" in contents and "第二轮：再补一句" in contents
    print(f"  第一轮后 messages 数：{first_len}")
    print(f"  第二轮后 messages 数：{len(contents)}")
    print(f"  第二轮读到第一轮内容：{'第一轮：我想试听' in contents}")
    print(f"  线程内累计轮次（agent 调用次数）：{len(agents['trial'].calls)}")
    return recovered and len(agents["trial"].calls) == 2


async def evidence_interrupt():
    print()
    print("=" * 68)
    print("证据 3：interrupt() 人工接管 + Command(resume=...) 恢复")
    print("=" * 68)
    from langgraph.types import Command

    classifier = DemoClassifier({}, default="trial")
    agents = _agents()
    graph = build_graph(classifier=classifier, agents=agents)
    cfg = {"configurable": {"thread_id": "demo-interrupt"}}

    await _collect(graph, new_state("我要转人工", "demo-interrupt"), cfg)
    snap = await graph.aget_state(cfg)
    pending = None
    for task in snap.tasks or ():
        if task.interrupts:
            pending = task.interrupts[0]
            break
    print(f"  第一轮暂停：pending interrupt = {pending.value if pending else None}")

    resumed = await _collect(graph, Command(resume="客服已介入"), cfg)
    joined = "".join(resumed)
    resumed_ok = "客服已介入" in joined
    final = await graph.aget_state(cfg)
    print(f"  Command(resume='客服已介入') -> {joined[:60]}")
    print(f"  最终状态 handoff = {final.values['handoff']}")
    return pending is not None and resumed_ok and final.values["handoff"] is True


async def main():
    r1 = await evidence_routing()
    r2 = await evidence_persistence()
    r3 = await evidence_interrupt()
    print()
    print("=" * 68)
    print(f"结论：路由={r1}  跨轮持久={r2}  interrupt 恢复={r3}")
    print("=" * 68)
    return 0 if (r1 and r2 and r3) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
