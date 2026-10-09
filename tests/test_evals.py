"""
EDD 评估体系测试: 全部 mock, 不依赖真实 LLM / 网络。

覆盖:
1. gates 判定: 阈值边界 (正好等于 / 略低 / 略高 / 指标缺失)
2. runner 编排: mock 六类 evaluate 函数, 断言报告汇总 + dry-run 结构校验
3. backfill: append 去重、新 id 生成、无效样本跳过
4. 评估集完整性: 六类 json 都能加载、字段齐全、id 唯一

约束 (与仓库其他测试一致): 无 pytest-asyncio; 异步用例用 _run 同步包装。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.gates import check_gates, load_gates  # noqa: E402
from eval.runner import (  # noqa: E402
    EDDRunner,
    _summarize,
    load_cases,
    percentile,
    validate_cases_dir,
)
from eval.backfill import backfill  # noqa: E402


# ===========================================================================
# 1. 门禁判定 (阈值边界)
# ===========================================================================

def _make_report(metrics_override=None, kind_override=None):
    """构造一份最小 report 供 check_gates 用。"""
    base_metrics = {
        "routing": {"accuracy": 0.9},
        "rag": {"recall_at_k": 0.9},
        "slot": {"accuracy": 0.9},
        "tool": {"accuracy": 0.9},
        "handoff": {"accuracy": 0.9},
        "safety": {"accuracy": 1.0},
    }
    if kind_override:
        base_metrics.update(kind_override)
    if metrics_override:
        base_metrics.update(metrics_override)
    return {
        "results": {k: {"metrics": v} for k, v in base_metrics.items()},
        "metrics": {
            "latency_p95_ms": 1000.0,
            "tokens": 10000,
            "max_steps": 3,
        },
    }


class TestCheckGates:
    def test_all_pass_when_above_thresholds(self):
        report = _make_report()
        gates = load_gates()
        check = check_gates(report, gates)
        assert check["passed"] is True
        assert check["failures"] == []

    def test_accuracy_exactly_equal_threshold_passes(self):
        """边界: min 门禁 正好等于阈值 → 通过 (>= 语义)。"""
        report = _make_report({"routing": {"accuracy": 0.85}})
        gates = load_gates()
        check = check_gates(report, gates)
        assert check["passed"] is True
        assert all(f["gate"] != "routing_min_accuracy" for f in check["failures"])

    def test_accuracy_slightly_below_threshold_fails(self):
        report = _make_report({"routing": {"accuracy": 0.849}})
        gates = load_gates()
        check = check_gates(report, gates)
        assert check["passed"] is False
        fail = [f for f in check["failures"] if f["gate"] == "routing_min_accuracy"]
        assert fail and fail[0]["actual"] == 0.849
        assert fail[0]["threshold"] == 0.85

    def test_max_latency_exactly_equal_passes(self):
        """边界: max 门禁 正好等于阈值 → 通过 (<= 语义)。"""
        report = _make_report()
        report["metrics"]["latency_p95_ms"] = 5000
        check = check_gates(report, load_gates())
        assert check["passed"] is True

    def test_max_latency_slightly_above_fails(self):
        report = _make_report()
        threshold = load_gates()["max_p95_latency_ms"]
        report["metrics"]["latency_p95_ms"] = threshold + 1
        check = check_gates(report, load_gates())
        assert check["passed"] is False
        fail = [f for f in check["failures"] if f["gate"] == "max_p95_latency_ms"]
        assert fail and fail[0]["actual"] == threshold + 1

    def test_safety_less_than_1_fails(self):
        """红队必须全拦截: 0.9 < 1.0 → 失败。"""
        report = _make_report({"safety": {"accuracy": 0.9}})
        check = check_gates(report, load_gates())
        assert check["passed"] is False
        fail = [f for f in check["failures"] if f["gate"] == "safety_min_pass_rate"]
        assert fail and fail[0]["actual"] == 0.9

    def test_missing_metric_fails_with_reason(self):
        """指标缺失 (如离线未产出某类) → 该 gate 判失败并注明路径。"""
        report = _make_report()
        del report["results"]["tool"]
        check = check_gates(report, load_gates())
        assert check["passed"] is False
        fail = [f for f in check["failures"] if f["gate"] == "tool_min_success"]
        assert fail and "reason" in fail[0]

    def test_unknown_gate_ignored(self):
        report = _make_report()
        gates = dict(load_gates())
        gates["not_a_gate"] = 0.5
        check = check_gates(report, gates)
        assert check["passed"] is True


# ===========================================================================
# 2. runner 编排 (mock 六类 evaluate)
# ===========================================================================

class TestRunnerOrchestration:
    def _mock_runner(self, monkeypatch):
        """把 EDDRunner 的六个 _evaluate_* 换成可控 mock。"""
        calls = []
        real_methods = {}

        def _make_fake(kind, accuracy):
            def fake(self):
                calls.append(kind)
                details = [{
                    "id": f"{kind}_x", "ok": True, "latency_ms": 10.0,
                    "tokens": 5, "steps": 1,
                    "expected": {}, "actual": {},
                }]
                return _summarize(kind, details, extra_metrics={
                    "recall_at_k": accuracy if kind == "rag" else None,
                })

            return fake

        for kind, acc in [("routing", 0.9), ("rag", 0.8), ("slot", 0.9),
                          ("tool", 0.9), ("handoff", 0.9), ("safety", 1.0)]:
            real_methods[kind] = getattr(EDDRunner, f"_evaluate_{kind}")
            monkeypatch.setattr(EDDRunner, f"_evaluate_{kind}", _make_fake(kind, acc))

        runner = EDDRunner(dry_run=False, llm=None)
        report = runner.run_all()
        return report, calls

    def test_run_all_invokes_six_kinds(self, monkeypatch):
        _, calls = self._mock_runner(monkeypatch)
        assert calls == ["routing", "rag", "slot", "tool", "handoff", "safety"]

    def test_report_structure_and_aggregation(self, monkeypatch):
        report, _ = self._mock_runner(monkeypatch)
        assert set(report["results"]) == {
            "routing", "rag", "slot", "tool", "handoff", "safety",
        }
        for kind, res in report["results"].items():
            assert res["passed"] == 1
            assert res["total"] == 1
            assert res["metrics"]["accuracy"] > 0
        # 全局指标聚合
        assert report["metrics"]["latency_p95_ms"] > 0
        assert report["metrics"]["tokens"] == 30  # 6 类 × 5
        assert report["metrics"]["max_steps"] == 1
        # 门禁判定已内嵌
        assert report["gates"]["check"]["passed"] is True
        # 口径说明
        assert "accuracy" in report["methodology"]
        assert "latency_p95" in report["methodology"]
        assert "tokens" in report["methodology"]

    def test_only_subset(self, monkeypatch):
        calls = []
        for kind in ("routing", "rag", "slot", "tool", "handoff", "safety"):
            def fake(self, kind=kind):
                calls.append(kind)
                return _summarize(kind, [{
                    "id": f"{kind}_x", "ok": True, "latency_ms": 1.0,
                    "tokens": 0, "steps": 0, "expected": {}, "actual": {},
                }])
            monkeypatch.setattr(EDDRunner, f"_evaluate_{kind}", fake)

        runner = EDDRunner(dry_run=False, llm=None, only=["routing", "safety"])
        report = runner.run_all()
        assert calls == ["routing", "safety"]
        assert set(report["results"]) == {"routing", "safety"}

    def test_dry_run_validates_structure(self):
        """dry-run 走 validate_cases_dir: 六类都能加载且非空。"""
        summary = validate_cases_dir()
        assert set(summary) == {"routing", "rag", "slot", "tool", "handoff", "safety"}
        for kind, info in summary.items():
            assert info["cases"] >= 5, f"{kind} 少于 5 条"
            assert info["version"]

    def test_dry_run_runner_produces_report(self):
        runner = EDDRunner(dry_run=True)
        report = runner.run_all()
        assert set(report["results"]) == set(runner.only)
        assert report["meta"]["dry_run"] is True
        # dry-run 的 case 标记 ok=None
        for res in report["results"].values():
            assert res["details"][0]["ok"] is None


# ===========================================================================
# 3. backfill (append 去重 + 新 id)
# ===========================================================================

class TestBackfill:
    def _write_tmp_set(self, tmp_path, kind="routing"):
        """复制一份评估集到 tmp 目录, 返回 (cases_dir, 原 case 数)。"""
        import shutil
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        src = PROJECT_ROOT / "eval" / "cases"
        shutil.copy(src / "routing.json", cases_dir / "routing.json")
        data = json.loads((cases_dir / "routing.json").read_text(encoding="utf-8"))
        return cases_dir, len(data["cases"])

    def test_append_new_case_with_autogen_id(self, tmp_path):
        cases_dir, n = self._write_tmp_set(tmp_path)
        result = backfill("routing", [{"input": "新样本", "expected_intent": "package"}],
                          cases_dir=cases_dir)
        assert len(result["added"]) == 1
        assert result["total"] == n + 1
        # 新 id 自动生成 routing_013 (现有最大 012)
        assert result["added"][0] == "routing_013"
        data = json.loads((cases_dir / "routing.json").read_text(encoding="utf-8"))
        assert len(data["cases"]) == n + 1
        assert data["cases"][-1]["input"] == "新样本"

    def test_dedupe_by_existing_id_skipped(self, tmp_path):
        cases_dir, n = self._write_tmp_set(tmp_path)
        result = backfill("routing", [{"id": "routing_001", "input": "重复",
                                       "expected_intent": "package"}],
                          cases_dir=cases_dir)
        assert result["added"] == []
        assert result["skipped_dup"] == ["routing_001"]
        assert result["total"] == n  # 未追加

    def test_invalid_sample_skipped(self, tmp_path):
        cases_dir, n = self._write_tmp_set(tmp_path)
        result = backfill("routing", [{"input": "缺 expected_intent"}],
                          cases_dir=cases_dir)
        assert result["added"] == []
        assert len(result["skipped_invalid"]) == 1
        assert result["total"] == n

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError):
            backfill("nope", [], cases_dir=tmp_path)

    def test_mixed_batch_dup_and_new(self, tmp_path):
        cases_dir, n = self._write_tmp_set(tmp_path)
        result = backfill("routing", [
            {"id": "routing_001", "input": "重复", "expected_intent": "package"},
            {"input": "合法新样本", "expected_intent": "unrelated"},
        ], cases_dir=cases_dir)
        assert result["added"] == ["routing_013"]
        assert result["skipped_dup"] == ["routing_001"]
        assert result["total"] == n + 1


# ===========================================================================
# 4. 评估集完整性
# ===========================================================================

class TestEvalSetIntegrity:
    def test_all_six_kinds_loadable(self):
        for kind in ("routing", "rag", "slot", "tool", "handoff", "safety"):
            data = load_cases(kind)
            assert len(data["cases"]) >= 5, f"{kind} 少于 5 条"

    def test_id_unique_across_all_kinds(self):
        for kind in ("routing", "rag", "slot", "tool", "handoff", "safety"):
            data = load_cases(kind)
            ids = [c["id"] for c in data["cases"]]
            assert len(ids) == len(set(ids)), f"{kind} id 重复"

    def test_required_fields_per_kind(self):
        required = {
            "routing": ["id", "version", "input", "expected_intent"],
            "rag": ["id", "version", "query", "expected_doc_ids"],
            "slot": ["id", "version", "utterance", "expected"],
            "tool": ["id", "version", "message", "expected_tool_called"],
            "handoff": ["id", "version", "turns", "expected_state_sequence"],
            "safety": ["id", "version", "input", "category"],
        }
        for kind, fields in required.items():
            data = load_cases(kind)
            for c in data["cases"]:
                missing = [f for f in fields if f not in c]
                assert not missing, f"{kind} {c['id']} 缺字段 {missing}"

    def test_routing_expected_intent_in_whitelist(self):
        data = load_cases("routing")
        whitelist = set(data["intent_values"])
        for c in data["cases"]:
            assert c["expected_intent"] in whitelist

    def test_handoff_turns_and_intents_aligned(self):
        data = load_cases("handoff")
        for c in data["cases"]:
            assert len(c["turns"]) == len(c.get("intents", []))
            assert len(c["turns"]) == len(c["expected_state_sequence"])

    def test_safety_category_in_whitelist(self):
        data = load_cases("safety")
        for c in data["cases"]:
            assert c["category"] in {"jailbreak", "harmful", "privacy"}

    def test_slot_expected_fields_valid(self):
        data = load_cases("slot")
        valid = {"subject", "grade_level", "start_time", "duration_minutes",
                 "preferred_teacher_name", "preferred_style", "info_complete"}
        for c in data["cases"]:
            assert valid.issuperset(c["expected"].keys())

    def test_rag_expected_doc_ids_reference_corpus(self):
        data = load_cases("rag")
        corpus_ids = {int(d["doc_id"]) for d in data["corpus"]}
        for c in data["cases"]:
            assert set(c["expected_doc_ids"]).issubset(corpus_ids)


# ===========================================================================
# 5. 工具函数
# ===========================================================================

class TestPercentile:
    def test_empty(self):
        assert percentile([], 95.0) == 0.0

    def test_single(self):
        assert percentile([42.0], 95.0) == 42.0

    def test_sorted_behavior(self):
        assert percentile([1, 2, 3, 4], 100.0) == 4
        assert percentile([1, 2, 3, 4], 0.0) == 1
        # nearest-rank: 50% of 4 = 2 → ceil=2 → 排序第 2 个 = 2
        assert percentile([1, 2, 3, 4], 50.0) == 2

    def test_p95_roughly(self):
        vals = list(range(1, 101))
        p = percentile(vals, 95.0)
        assert p == 95  # nearest-rank: ceil(0.95*100)=95 → 第 95 个 = 95
