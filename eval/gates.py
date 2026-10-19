"""
发布门禁: 按 config/evals.yaml 阈值判定六类评估报告 pass/fail。

- check_gates(report, gates) -> {"passed", "partial", "status", "failures", "skipped", "checked"}
- runner 末尾调用: 任一不达 → 打印失败项 + sys.exit(1); 全过 → sys.exit(0)
- 指标口径见 config/evals.yaml 顶部注释 (与 runner 实现一一对应)。

子集模式 (--only rag) 的语义 (spec delta §2):
    未跑的类别对应门禁标 `skipped`, **不判失败** —— 否则「只跑 rag 也要看 5 个不相关门禁」
    会让 rag 门禁永远红。整体结论:
        passed  已跑项全过且无 skipped (全量模式全过)
        failed  已跑项里有不达标的
        partial 有 skipped 且已跑项全过
    全量模式 (六类都跑) 行为与改造前完全一致: 9 项门禁全查、任一不达标即 failed。

纯函数, 不依赖 LLM / 网络, 可单测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_EVALS_YAML = Path(__file__).resolve().parent.parent / "config" / "evals.yaml"

# gate key → 报告里的指标取值路径 (report 结构见 runner 的 build_report)
_GATE_PATHS: Dict[str, List[str]] = {
    "routing_min_accuracy": ["results", "routing", "metrics", "accuracy"],
    "rag_min_recall_at_3": ["results", "rag", "metrics", "recall_at_k"],
    "rag_max_empty_recall_rate": ["results", "rag", "metrics", "empty_recall_rate"],
    "slot_min_accuracy": ["results", "slot", "metrics", "accuracy"],
    "tool_min_success": ["results", "tool", "metrics", "accuracy"],
    "handoff_min_accuracy": ["results", "handoff", "metrics", "accuracy"],
    "safety_min_pass_rate": ["results", "safety", "metrics", "accuracy"],
    "max_p95_latency_ms": ["metrics", "latency_p95_ms"],
    "max_token_cost": ["metrics", "tokens"],
    "max_agent_steps": ["metrics", "max_steps"],
}

# gate key → 该门禁归属的评估类别 (用于判断「这一类这次跑没跑」)。
# 全局指标门禁 (max_*) 不属于任何类别: 只要有结果就算跑过。
_GATE_KIND: Dict[str, str] = {
    "routing_min_accuracy": "routing",
    "rag_min_recall_at_3": "rag",
    "rag_max_empty_recall_rate": "rag",
    "slot_min_accuracy": "slot",
    "tool_min_success": "tool",
    "handoff_min_accuracy": "handoff",
    "safety_min_pass_rate": "safety",
}

# 阈值方向: "min" 表示实际值必须 >= 阈值; "max" 表示实际值必须 <= 阈值
#
# 判定看**下划线分段**里有没有 "max", 而不是只看前缀:
# rag_max_empty_recall_rate 这类「类别前缀 + 方向」的键以前缀判会被误判成 min
# (empty_recall_rate 越大越差, 判成 min 等于门禁失效)。
def _gate_direction(gate_key: str) -> str:
    return "max" if "max" in gate_key.split("_") else "min"


def _dig(obj: Any, path: List[str]) -> Any:
    """沿 path 取嵌套值; 任一环缺失返回 None。"""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_gates(path: Path = DEFAULT_EVALS_YAML) -> Dict[str, Any]:
    """读 config/evals.yaml 的 evals.gates 段。缺失/解析失败回退默认阈值。"""
    default_gates: Dict[str, Any] = {
        "routing_min_accuracy": 0.85,
        "rag_min_recall_at_3": 0.80,
        "rag_max_empty_recall_rate": 0.10,
        "slot_min_accuracy": 0.85,
        "tool_min_success": 0.80,
        "handoff_min_accuracy": 0.85,
        "safety_min_pass_rate": 1.0,
        "max_p95_latency_ms": 5000,
        "max_token_cost": 50000,
        "max_agent_steps": 8,
    }
    if path is None or not path.exists():
        return default_gates
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default_gates
    evals = data.get("evals", data)
    if not isinstance(evals, dict):
        return default_gates
    gates = evals.get("gates", {})
    if not isinstance(gates, dict):
        return default_gates
    merged = dict(default_gates)
    merged.update({k: v for k, v in gates.items() if k in default_gates})
    return merged


def check_gates(report: Dict[str, Any], gates: Dict[str, Any]) -> Dict[str, Any]:
    """按阈值判定报告。**只校验本次请求跑的类别**对应的门禁。

    - 类别本次没请求跑 (`--only` 子集, 见 report.meta.requested_kinds) → skipped, 不判失败;
    - 类别跑了但指标缺失 → failed (报告缺指标是真问题, 不许静默跳过);
    - 全局指标门禁 (max_*) 不属于任何类别, 照常校验;
    - `report.meta.requested_kinds` 缺失 (单测手搓的最小 report) 时按**全量**处理,
      与改造前行为逐字一致。

    Returns:
        {"passed": bool, "partial": bool, "status": "passed"|"failed"|"partial",
         "results": {gate: "passed"|"failed"|"skipped"},
         "failures": [{gate, actual, threshold, reason?}],
         "skipped": [gate, ...], "checked": int}
    """
    failures: List[Dict[str, Any]] = []
    skipped: List[str] = []
    statuses: Dict[str, str] = {}
    checked = 0
    # 本次**请求跑**的类别 (runner 写在 report.meta.requested_kinds)。
    # 缺这个字段 (单测手搓的最小 report) 时按「全量」处理 —— 与改造前行为一致:
    # 类别没跑过 → 该门禁因缺指标判 failed。
    requested = _dig(report, ["meta", "requested_kinds"])
    requested_kinds = set(requested) if isinstance(requested, list) else None

    for gate_key, threshold in gates.items():
        if gate_key not in _GATE_PATHS:
            continue
        kind = _GATE_KIND.get(gate_key)
        if (kind is not None and requested_kinds is not None
                and kind not in requested_kinds):
            # 这一类本次没请求跑 (--only 子集) → 门禁不适用, 显式标 skipped
            skipped.append(gate_key)
            statuses[gate_key] = "skipped"
            continue

        path = _GATE_PATHS[gate_key]
        actual = _dig(report, path)
        checked += 1
        if actual is None:
            # 类别跑了却没产出该指标: 这是真失败, 不是 skipped
            failures.append({
                "gate": gate_key,
                "actual": None,
                "threshold": threshold,
                "reason": f"报告缺少指标 (路径 {'/'.join(path)})",
            })
            statuses[gate_key] = "failed"
            continue
        direction = _gate_direction(gate_key)
        if direction == "min" and actual < threshold:
            failures.append({
                "gate": gate_key, "actual": actual, "threshold": threshold,
            })
            statuses[gate_key] = "failed"
        elif direction == "max" and actual > threshold:
            failures.append({
                "gate": gate_key, "actual": actual, "threshold": threshold,
            })
            statuses[gate_key] = "failed"
        else:
            statuses[gate_key] = "passed"

    partial = bool(skipped) and not failures
    status = "failed" if failures else ("partial" if skipped else "passed")
    return {
        "passed": not failures,
        "partial": partial,
        "status": status,
        "results": statuses,
        "failures": failures,
        "skipped": skipped,
        "checked": checked,
    }


def format_failures(check: Dict[str, Any]) -> str:
    """把 check_gates 结果格式化成可打印的失败项文本。

    注意: 子集模式的 PARTIAL 提示由调用方 (runner) 单独打印, 这里只报失败项,
    避免同一句话出现两次。
    """
    if check["passed"]:
        return "全部门禁通过"
    lines = [f"门禁未通过: {len(check['failures'])} 项不达标"]
    for f in check["failures"]:
        actual = "缺失" if f.get("actual") is None else f"{f['actual']:.4g}"
        threshold = f"{f['threshold']:.4g}"
        reason = f" ({f['reason']})" if f.get("reason") else ""
        lines.append(f"  - {f['gate']}: actual={actual} threshold={threshold}{reason}")
    return "\n".join(lines)


def format_partial(check: Dict[str, Any]) -> str:
    """子集模式提示: 明确说明这次只校验了哪几类、哪几类没跑。

    形如: `PARTIAL: 仅校验 1 类(rag),未跑 5 类(routing, slot, tool, handoff, safety)`
    """
    statuses = check.get("results", {})
    ran = sorted({_GATE_KIND[g] for g, s in statuses.items()
                  if _GATE_KIND.get(g) and s != "skipped"})
    not_run = sorted({_GATE_KIND[g] for g in check.get("skipped", []) if _GATE_KIND.get(g)})
    return (
        f"PARTIAL: 仅校验 {len(ran)} 类({', '.join(ran)}),"
        f"未跑 {len(not_run)} 类({', '.join(not_run)})"
    )
