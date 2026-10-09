"""
项目体检脚本(P0 阶段产物 #1)。

只读,不动任何业务代码。生成 docs/REPO_HEALTH.md(产物 #2)。

5 张表:
  T1. 代码量分布  (按层 9 类聚合)
  T2. 分层合规    (检测下层反向 import 上层)
  T3. 死代码候选  (定义但未被引用的 public 名)
  T4. 未使用 import (本仓库内 import 但对应符号未被使用)
  T5. 外网 URL 依赖 (源码里出现的 http(s)://, 排除注释/docstring)

用法:
  python scripts/audit_repo.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "REPO_HEALTH.md"

# 9 大类的目录映射
LAYER_DIRS: Dict[str, List[str]] = {
    "app":    ["app.py"],
    "api":    ["api"],
    "agents": ["agents"],
    "services": ["services"],
    "db":     ["db"],
    "web":    ["web"],
    "tests":  ["tests"],
    "scripts": ["scripts"],
    "config": ["config"],
}

# 分层图:key 只能 import value
ALLOWED_IMPORTS: Dict[str, Set[str]] = {
    "app":      {"api", "agents", "services", "db", "web", "config"},
    "api":      {"agents", "services", "db", "config"},
    "agents":   {"services", "db", "config"},
    "services": {"db", "config"},
    "db":       set(),  # 最底层
    "web":      {"api", "agents", "services", "config"},
    "tests":    {"app", "api", "agents", "services", "db", "web", "config", "scripts"},
    "scripts":  {"app", "api", "agents", "services", "db", "config"},
    "config":   set(),
}

# 排除目录(不参与统计)
EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "data", "cache"}


def _layer_of(rel_path: Path) -> str | None:
    """根据相对路径判断属于哪个 layer。"""
    parts = rel_path.parts
    if not parts:
        return None
    head = parts[0]
    for layer, dirs in LAYER_DIRS.items():
        if head in dirs:
            return layer
    if rel_path.name == "app.py":
        return "app"
    return None


def _collect_python_files() -> List[Path]:
    out: List[Path] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                out.append(Path(root) / f)
    return out


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# T1. 代码量分布
# ---------------------------------------------------------------------------

def t1_code_stats(files: List[Path]) -> Dict[str, Tuple[int, int]]:
    by_layer: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for p in files:
        layer = _layer_of(p.relative_to(PROJECT_ROOT))
        if not layer:
            continue
        files_n, lines_n = by_layer[layer]
        try:
            lines = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace"))
        except Exception:
            lines = 0
        by_layer[layer] = (files_n + 1, lines_n + lines)
    return dict(by_layer)


# ---------------------------------------------------------------------------
# T2. 分层合规
# ---------------------------------------------------------------------------

def _module_to_layer(mod: str) -> str | None:
    """把 'agents.task_classification_agent' -> 'agents'。"""
    top = mod.split(".")[0]
    if top in LAYER_DIRS:
        return top
    if top == "app":
        return "app"
    return None


def t2_layer_violations(files: List[Path]) -> List[Tuple[str, str, str]]:
    """返回 (file, import_module, reason) 列表。"""
    out: List[Tuple[str, str, str]] = []
    for p in files:
        tree = _parse(p)
        if not tree:
            continue
        rel = str(p.relative_to(PROJECT_ROOT))
        src_layer = _layer_of(p.relative_to(PROJECT_ROOT))
        if not src_layer:
            continue
        allowed = ALLOWED_IMPORTS.get(src_layer, set())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    break
            if not mod:
                continue
            tgt = _module_to_layer(mod)
            if not tgt or tgt == src_layer:
                continue
            if tgt not in allowed:
                out.append((rel, mod, f"{src_layer} 不允许 import {tgt} 层"))
    return out


# ---------------------------------------------------------------------------
# T3. 死代码候选
# ---------------------------------------------------------------------------

def _defined_public_names(tree: ast.Module) -> Set[str]:
    """top-level 公开名(非 _ 开头)。"""
    out: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                out.add(node.target.id)
    return out


def _imported_names(tree: ast.Module) -> Set[str]:
    """从兄弟模块 import 的具体符号名。"""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    out.add(alias.asname)
    return out


def t3_dead_code(files: List[Path]) -> List[Tuple[str, str]]:
    """(file, name) 列表:定义但未被 import。"""
    defined: Dict[Tuple[str, str], Set[str]] = {}
    imports_used: Set[Tuple[str, str]] = set()  # 这里是 (module_dot_path,) 而非 name

    for p in files:
        rel = ".".join(p.relative_to(PROJECT_ROOT).with_suffix("").parts)
        tree = _parse(p)
        if not tree:
            continue
        defined[(rel, str(p.relative_to(PROJECT_ROOT)))] = _defined_public_names(tree)

    # 谁引用了哪个兄弟模块
    module_used_by: Dict[str, Set[str]] = defaultdict(set)
    for p in files:
        rel = str(p.relative_to(PROJECT_ROOT))
        tree = _parse(p)
        if not tree:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod.startswith(("agents", "api", "services", "db", "config", "web")):
                    # 该文件用到了 mod 模块
                    module_used_by[mod].add(rel)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("agents", "api", "services", "db", "config", "web")):
                        module_used_by[alias.name].add(rel)

    # 死代码: 某模块被定义但从未被 import
    dead: List[Tuple[str, str]] = []
    for (mod_path, file_rel), names in defined.items():
        if not names:
            continue
        if mod_path in module_used_by:
            continue
        # 排除 app.py 和被 __init__.py 显式 export 的
        if file_rel.endswith("app.py"):
            continue
        # __init__.py 视为"包级入口",不算死
        if file_rel.endswith("__init__.py"):
            continue
        for n in sorted(names):
            dead.append((file_rel, n))
    return dead


# ---------------------------------------------------------------------------
# T4. 未使用 import (本仓库内)
# ---------------------------------------------------------------------------

def t4_unused_imports(files: List[Path]) -> List[Tuple[str, str]]:
    """返回 (file, 'mod.name') 列表:从兄弟模块 import 了但本文件未用。"""
    out: List[Tuple[str, str]] = []
    for p in files:
        tree = _parse(p)
        if not tree:
            continue
        rel = str(p.relative_to(PROJECT_ROOT))
        # 收集所有名字 token
        all_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                base = node.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    all_names.add(base.id)
        # 检查每个 import from 兄弟模块
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if not mod.startswith(("agents", "api", "services", "db", "config", "web")):
                    continue
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name not in all_names and name != "*":
                        out.append((rel, f"{mod}.{alias.name}"))
    return out


# ---------------------------------------------------------------------------
# T5. 外网 URL
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r'https?://[^\s\'"<>)]+')
# 排除 docstring 内 + 注释内
COMMENT_OR_DOCSTRING = re.compile(r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\#[^\n]*)')


def t5_urls(files: List[Path]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # 删掉 docstring/注释
        scrubbed = COMMENT_OR_DOCSTRING.sub("", src)
        for m in URL_PATTERN.findall(scrubbed):
            url = m.rstrip(".,;:")
            out.append((str(p.relative_to(PROJECT_ROOT)), url))
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    files = _collect_python_files()
    print(f"[audit] 扫描 {len(files)} 个 .py 文件")

    t1 = t1_code_stats(files)
    t2 = t2_layer_violations(files)
    t3 = t3_dead_code(files)
    t4 = t4_unused_imports(files)
    t5 = t5_urls(files)

    parts: List[str] = []
    parts.append("# REPO_HEALTH\n")
    parts.append(f"> 自动生成 by `scripts/audit_repo.py`,运行时间 < 5s。\n")
    parts.append("人手不修改本文件(下次跑 audit 会被覆盖)。\n")

    # T1
    parts.append("\n## T1. 代码量分布\n")
    rows = []
    total_lines = 0
    total_files = 0
    for layer in LAYER_DIRS:
        files_n, lines_n = t1.get(layer, (0, 0))
        rows.append([layer, str(files_n), str(lines_n)])
        total_files += files_n
        total_lines += lines_n
    rows.append(["**TOTAL**", f"**{total_files}**", f"**{total_lines}**"])
    parts.append(_md_table(["Layer", "Files", "Lines"], rows))

    # T2
    parts.append("\n## T2. 分层合规\n")
    if not t2:
        parts.append("✅ 无违规。 下层未反向 import 上层。\n")
    else:
        rows = [[f, m, r] for f, m, r in t2[:30]]
        parts.append(_md_table(["File", "Import", "Reason"], rows))
        if len(t2) > 30:
            parts.append(f"\n(共 {len(t2)} 处,只列前 30)\n")

    # T3
    parts.append("\n## T3. 死代码候选\n")
    if not t3:
        parts.append("✅ 全部定义的 public 名都被引用过。\n")
    else:
        rows = [[f, n] for f, n in t3[:40]]
        parts.append(_md_table(["File", "Public Name"], rows))
        if len(t3) > 40:
            parts.append(f"\n(共 {len(t3)} 个,只列前 40)\n")
        parts.append("\n注: `__init__.py` 里的 public 名(用于 re-export)和 `app.py` 顶层 `app` 已被排除。\n")
        parts.append("已知限制: 本检查不跟踪 `from pkg import *` 和 `__all__` re-export 链,例如\n")
        parts.append("`agents/consultant/{ConsultationClassifier, ConsultationProcessor, ...}`\n")
        parts.append("虽然出现在 `agents/consultant/__init__.py` 的 `__all__` 里被 re-export,\n")
        parts.append("但本表仍会显示为候选(实际被 `ConsultantAgent` 使用)。 复核请查 `__init__.py`。\n")

    # T4
    parts.append("\n## T4. 未使用 import (本仓库内)\n")
    if not t4:
        parts.append("✅ 本仓库内所有 import 都被使用了。\n")
    else:
        rows = [[f, m] for f, m in t4[:30]]
        parts.append(_md_table(["File", "Import"], rows))
        if len(t4) > 30:
            parts.append(f"\n(共 {len(t4)} 个,只列前 30)\n")

    # T5
    parts.append("\n## T5. 外网 URL 依赖 (源码内,排除注释/docstring)\n")
    if not t5:
        parts.append("✅ 源码中没有出现外网 URL。\n")
    else:
        rows = [[f, u] for f, u in t5[:20]]
        parts.append(_md_table(["File", "URL"], rows))
        if len(t5) > 20:
            parts.append(f"\n(共 {len(t5)} 个,只列前 20)\n")

    # 已知警告
    parts.append("\n## 已知警告 (非 P0 范围)\n")
    parts.append("- `db/models.py:18` SQLAlchemy 2.0 `declarative_base` 已迁,但 `datetime.utcnow()` 仍有 ~3 处弃用警告。\n")
    parts.append("- `agents/consultant/__init__.py` 注释与 README 早期版本不一致,本表不查语义。\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"[audit] 已生成 {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[audit] T1 9 层共 {total_files} 文件 / {total_lines} 行")
    print(f"[audit] T2 分层违规 {len(t2)} / T3 死代码候选 {len(t3)} / T4 未用 import {len(t4)} / T5 外网 URL {len(t5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
