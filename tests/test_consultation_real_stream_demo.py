"""独立进程的真实 HTTP socket 屏障验真，不使用 TestClient/ASGITransport。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo_consultation_real_stream.py"


def test_offline_http_stream_demo_is_repeatable_and_disposable():
    assert SCRIPT.is_file(), "真实 HTTP 咨询流 demo 尚未实现"
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT,
            env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUTF8": "1"},
            capture_output=True, text=True, encoding="utf-8", timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "FIRST_BODY_BEFORE_MODEL_DONE=True" in result.stdout
        reports = [line.removeprefix("STREAM_DEMO_RESULT ") for line in result.stdout.splitlines()
                   if line.startswith("STREAM_DEMO_RESULT ")]
        assert len(reports) == 1, result.stdout
        report = json.loads(reports[0])
        assert report["transport"] == "loopback_tcp_http"
        assert report["first_body_before_model_done"] is True
        assert report["complete_body"] is True
        assert report["midstream_partial"] is True
        assert report["cancel_cleanup"] is True
        assert report["server_stopped"] is True
        assert report["paths"] == ["/chat/stream", "/chat"]
        assert not Path(report["temporary_directory"]).exists()
