"""
回灌闭环: 从线上失败样本回灌评估集 (追加新 case, 按 id 去重)。

- `python -m eval.backfill --type routing --file failures.json`
  读失败样本 (JSON 数组或 {"cases": [...]}) → 按类型 append 到 eval/cases/<type>.json
- 去重: 已存在同 id 跳过; 新 id 自动生成 (<type>_<nnn>, 取现有最大序号 +1)
- 从 observability 的失败 trace 日志手动挑样本: trace 日志里 span error=true 的
  case 摘出 input/expected 后存成 JSON 文件再回灌 (docstring 说明, 不自动解析日志)。

回灌样本字段:
  - routing:  {input, expected_intent, note?}
  - rag:      {query, expected_doc_ids, note?}
  - slot:     {utterance, expected: {...}, note?}
  - tool:     {message, teacher_id?, time_iso?, subject?, expected_tool_called, note?}
  - handoff:  {turns: [...], intents: [...], expected_state_sequence: [...], note?}
  - safety:   {input, category, note?}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import CASE_KINDS, CASES_DIR, load_cases

# 各类型的"期望字段"要求 (回灌样本必须具备, 否则跳过并提示)
_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "routing": ["input", "expected_intent"],
    "rag": ["query", "expected_doc_ids"],
    "slot": ["utterance", "expected"],
    "tool": ["message", "expected_tool_called"],
    "handoff": ["turns", "expected_state_sequence"],
    "safety": ["input", "category"],
}


def _next_id(kind: str, existing_ids: List[str]) -> str:
    """生成新 id: <kind>_<max(现有序号)+1>。"""
    max_num = 0
    prefix = f"{kind}_"
    for cid in existing_ids:
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", str(cid))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}{max_num + 1:03d}"


def _normalize_samples(raw: Any) -> List[Dict[str, Any]]:
    """把文件内容归一成样本列表: 支持顶层数组 或 {"cases": [...]}。"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("cases"), list):
        return raw["cases"]
    return []


def backfill(kind: str, samples: List[Dict[str, Any]],
             cases_dir: Path = CASES_DIR) -> Dict[str, Any]:
    """把失败样本回灌到 <type>.json, 返回 {added, skipped_dup, skipped_invalid, total}。

    不修改任何已存在 case (冻结评估集只增不改); 新样本追加到文件尾部,
    并把文件顶层 version 保持不变 (新 case 各自带 version 字段)。
    """
    if kind not in CASE_KINDS:
        raise ValueError(f"未知评估类: {kind!r}. 可选: {sorted(CASE_KINDS)}")

    path = cases_dir / CASE_KINDS[kind]
    data = load_cases(kind, cases_dir)
    cases = data["cases"]
    existing_ids = {str(c["id"]) for c in cases}

    required = _REQUIRED_FIELDS[kind]
    added: List[str] = []
    skipped_dup: List[str] = []
    skipped_invalid: List[str] = []

    for s in samples:
        sid = str(s.get("id", "")).strip()
        if sid and sid in existing_ids:
            skipped_dup.append(sid)
            continue
        if not all(s.get(f) is not None for f in required):
            missing = [f for f in required if s.get(f) is None]
            skipped_invalid.append(f"缺字段 {missing} -> {s}")
            continue
        new_case = dict(s)
        if not new_case.get("id"):
            new_case["id"] = _next_id(kind, [c["id"] for c in cases] + added)
        if "version" not in new_case:
            new_case["version"] = data.get("version", "1.0.0")
        cases.append(new_case)
        existing_ids.add(str(new_case["id"]))
        added.append(str(new_case["id"]))

    data["cases"] = cases
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "kind": kind,
        "added": added,
        "skipped_dup": skipped_dup,
        "skipped_invalid": skipped_invalid,
        "total": len(cases),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.backfill", description="失败样本回灌评估集")
    p.add_argument("--type", required=True, choices=sorted(CASE_KINDS), help="评估类型")
    p.add_argument("--file", required=True, help="失败样本 JSON 文件 (数组 或 {cases:[...]})")
    p.add_argument("--cases-dir", type=str, default=str(CASES_DIR), help="评估集目录")
    args = p.parse_args(argv)

    fpath = Path(args.file)
    if not fpath.exists():
        print(f"[backfill] 文件不存在: {fpath}", file=sys.stderr)
        return 2
    raw = json.loads(fpath.read_text(encoding="utf-8"))
    samples = _normalize_samples(raw)
    if not samples:
        print("[backfill] 文件中没有可回灌的样本", file=sys.stderr)
        return 2

    result = backfill(args.type, samples, Path(args.cases_dir))
    print(f"[backfill] type={result['kind']} 新增 {len(result['added'])} "
          f"去重跳过 {len(result['skipped_dup'])} 无效跳过 {len(result['skipped_invalid'])} "
          f"当前总数 {result['total']}")
    if result["added"]:
        print(f"  新增 id: {', '.join(result['added'])}")
    for invalid in result["skipped_invalid"]:
        print(f"  [无效] {invalid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
