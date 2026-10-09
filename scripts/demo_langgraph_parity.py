"""LangGraph P0-B 双跑对拍实跑演示（离线，无外部依赖，exit 0）。

对 `.claude/tasks/langgraph-migrate-p0b.md` 第二节：把 `tests/test_langgraph_parity.py`
里的每条多轮对话脚本在 legacy（TaskClassificationAgent）与 graph（build_graph）
两条链路上各跑一遍，打印逐轮输出对比与总体一致率。

- 两条链路注入同一套替身（同一分类器实例 + 同规格业务 agent / handoff / availability）；
- 比较前规范化剔除已知可接受差异行 `[THOUGHT][LangGraph] 意图识别为 X。`，其余逐字比较；
- 一致率 100% 时 exit 0，否则 exit 1。

运行：python scripts/demo_langgraph_parity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_langgraph_parity as parity  # noqa: E402


def _print_turn(turn):
    mark = "OK  " if turn.matched else "FAIL"
    print(f"    [{mark}] 轮 {turn.turn_index} 输入: {turn.user_input}")
    print(f"           legacy: {turn.legacy_normalized!r}")
    print(f"           graph : {turn.graph_normalized!r}")
    if not turn.matched:
        print(f"           graph(raw): {turn.graph_raw!r}")
        print(f"           legacy(raw): {turn.legacy_raw!r}")


def main():
    print("=" * 78)
    print("LangGraph P0-B 双跑对拍：legacy vs graph 逐字等价性")
    print("规范化剔除：'[THOUGHT][LangGraph] 意图识别为 X。'（已知可接受差异）")
    print("=" * 78)

    results = parity.run_parity()
    for result in results:
        print(f"\n[{result.scenario_id}] {result.name}  —  {result.note}")
        for turn in result.turns:
            _print_turn(turn)

    total = sum(r.total for r in results)
    matched = sum(r.matched for r in results)
    rate = parity.consistency_rate(results)

    print("\n" + "=" * 78)
    print(f"脚本条数: {len(results)}（含 spec 14 类场景 + {len(results) - 14} 条续轮态扩展）")
    print(f"比较轮数: {total}    命中轮数: {matched}")
    print(f"一致率  : {matched}/{total} = {rate:.4f}")
    failing = [r for r in results if r.matched != r.total]
    if failing:
        print("残留差异:")
        for result in failing:
            for turn in result.turns:
                if not turn.matched:
                    print(f"  - [{result.scenario_id}] 轮 {turn.turn_index}: "
                          f"legacy={turn.legacy_normalized!r} graph={turn.graph_normalized!r}")
    else:
        print("残留差异: 无")
    print("结论    : " + ("PASS（一致率 100%）" if rate == 1.0 else "FAIL"))
    print("=" * 78)
    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
