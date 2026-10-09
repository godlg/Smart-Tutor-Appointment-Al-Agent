"""
前端模板烟雾测试

目的: 防止 `web/templates/*.html` 的 `<script>` 块在重构时被改坏。
本测试只做 parse 验证 (即: 模板字面量、对象字面量、占位符语法合法),
不执行运行时逻辑。

依赖: 系统装了 `node` (任意版本),用于 vm.Script 解析。
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"


def _has_node() -> bool:
    return shutil.which("node") is not None


pytestmark = pytest.mark.skipif(not _has_node(), reason="node not available")


def _parse_with_node(script_body: str) -> tuple[bool, str]:
    """用 Node 的 vm.Script 解析一段 JS,返回 (success, error_msg)。"""
    body_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    )
    body_file.write(script_body)
    body_file.close()
    body_path = str(Path(body_file.name).resolve())

    runner_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    )
    runner_file.write(
        "const fs = require('fs');\n"
        "const vm = require('vm');\n"
        "const body = fs.readFileSync(process.argv[2], 'utf-8');\n"
        "const sb = { document: { getElementById: () => ({ innerHTML: '' }) }, fetch: () => {}, console };\n"
        "vm.createContext(sb);\n"
        "try { new vm.Script(body); console.log('OK'); }\n"
        "catch (e) { console.error('FAIL ' + e.message); process.exit(1); }\n"
    )
    runner_file.close()
    runner_path = str(Path(runner_file.name).resolve())

    try:
        r = subprocess.run(
            ["node", runner_path, body_path],
            capture_output=True, text=True, timeout=20,
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode == 0 and "OK" in r.stdout:
            return True, ""
        return False, (r.stderr or r.stdout).strip()[:300]
    finally:
        for p in (body_path, runner_path):
            try:
                Path(p).unlink()
            except OSError:
                pass


def _extract_scripts(html: str) -> list[tuple[str, str]]:
    """提取 HTML 中所有 <script>...</script> 块,返回 (label, body) 列表。"""
    out = []
    for i, m in enumerate(re.finditer(r"<script(?![^>]*src=)[^>]*>([\s\S]*?)</script>", html)):
        out.append((f"script#{i}", m.group(1)))
    return out


# 哪些模板里的 <script> 必须 parse 通过
# (只对含内联 JS 的页面做;纯展示页面不放进来)
SCRIPTS_TO_PARSE = [
    "lesson_series.html",   # 修过的那一页
    "packages.html",        # 修过的购买按钮
    "schedule_calendar.html",
    "index.html",
    "student_profile.html",
    "parent_center.html",
    "progress_dashboard.html",
    "teacher_list.html",
    "teacher_detail.html",
    "knowledge_management.html",
    "user_behavior_analysis.html",
]


@pytest.mark.parametrize("filename", SCRIPTS_TO_PARSE)
def test_template_script_parses(filename: str):
    """每个含内联 JS 的模板都应该能被 Node 解析。"""
    path = TEMPLATES_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} 不存在")
    html = path.read_text(encoding="utf-8")
    scripts = _extract_scripts(html)
    if not scripts:
        pytest.skip(f"{filename} 没有内联 <script>")
    for label, body in scripts:
        ok, err = _parse_with_node(body)
        assert ok, f"{filename} {label} parse failed: {err}"
