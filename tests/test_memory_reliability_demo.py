"""Run the standalone offline demo twice, outside pytest's in-process mocks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "demo_memory_reliability.py"


def test_offline_memory_demo_is_repeatable_and_recovers_in_fresh_process() -> None:
    assert SCRIPT.is_file(), "The memory reliability demo has not been implemented"
    for _ in range(2):
        process = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUTF8": "1"},
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        reports = [line.removeprefix("DEMO_RESULT ") for line in process.stdout.splitlines()
                   if line.startswith("DEMO_RESULT ")]
        assert len(reports) == 1, process.stdout
        report = json.loads(reports[0])
        assert report["checks"] == [
            "offline_guards", "scope_isolation", "durable_rollups",
            "archive_failure", "hard_budget", "real_consultation",
            "fresh_process_recovery",
        ]
        assert report["watermarks"] == [4, 8, 12]
        assert report["child_pid"] != report["parent_pid"]
        assert report["checkpoint_policy"] == "allowlisted_draft_safe_invalidation"
        assert not Path(report["temporary_directory"]).exists()
