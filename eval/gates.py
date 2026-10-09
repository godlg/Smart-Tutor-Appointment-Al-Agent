"""
发布门禁: 按 config/evals.yaml 阈值判定六类评估报告 pass/fail。

- check_gates(report, gates) -> {"passed": bool, "failures": [{gate, actual, threshold}]}
- runner 末尾调用: 任一不达 → 打印失败项 + sys.exit(1); 全过 → sys.exit(0)
- 指标口径见 config/evals.yaml 顶部注释 (与 runner 实现一一对应)。

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
    "slot_min_accuracy": ["results", "slot", "metrics", "accuracy"],
    "tool_min_success": ["results", "tool", "metrics", "accuracy"],
    "handoff_min_accuracy": ["results", "handoff", "metrics", "accuracy"],
    "safety_min_pass_rate": ["results", "safety", "metrics", "accuracy"],
    "max_p95_latency_ms": ["metrics", "latency_p95_ms"],
    "max_token_cost": ["metrics", "tokens"],
    "max_agent_steps": ["metrics", "max_steps"],
}

# 阈值方向: "min" 表示实际值必须 >= 阈值; "max" 表示实际值必须 <= 阈值
def _gate_direction(gate_key: str) -> str:
    return "max" if gate_key.startswith("max_") else "min"


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
    """按阈值判定报告。只检查 report 中实际存在指标的 gate;
    gate 对应指标缺失 (如某类评估因离线未产出) 时判为失败并注明原因。

    Returns:
        {"passed": bool, "failures": [{"gate", "actual", "threshold"}], "checked": int}
    """
    failures: List[Dict[str, Any]] = []
    checked = 0
    for gate_key, threshold in gates.items():
        if gate_key not in _GATE_PATHS:
            continue
        path = _GATE_PATHS[gate_key]
        actual = _dig(report, path)
        checked += 1
        if actual is None:
            failures.append({
                "gate": gate_key,
                "actual": None,
                "threshold": threshold,
                "reason": f"报告缺少指标 (路径 {'/'.join(path)})",
            })
            continue
        direction = _gate_direction(gate_key)
        if direction == "min" and actual < threshold:
            failures.append({
                "gate": gate_key, "actual": actual, "threshold": threshold,
            })
        elif direction == "max" and actual > threshold:
            failures.append({
                "gate": gate_key, "actual": actual, "threshold": threshold,
            })

    return {
        "passed": not failures,
        "failures": failures,
        "checked": checked,
    }


def format_failures(check: Dict[str, Any]) -> str:
    """把 check_gates 结果格式化成可打印的失败项文本。"""
    if check["passed"]:
        return "全部门禁通过"
    lines = [f"门禁未通过: {len(check['failures'])} 项不达标"]
    for f in check["failures"]:
        actual = "缺失" if f.get("actual") is None else f"{f['actual']:.4g}"
        threshold = f"{f['threshold']:.4g}"
        reason = f" ({f['reason']})" if f.get("reason") else ""
        lines.append(f"  - {f['gate']}: actual={actual} threshold={threshold}{reason}")
    return "\n".join(lines)
