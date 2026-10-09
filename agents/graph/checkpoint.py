"""LangGraph checkpointer 工厂（P1：SqliteSaver 持久化）。

默认仍 `InMemorySaver`（进程内）；`persistent=True` 时用 `AsyncSqliteSaver`，
落到项目既有 SQLite 库 —— checkpointer 另建 `checkpoints` / `writes` 表，
与既有业务表**同库共存**，不修改 / 删除任何既有表。

为什么是 `AsyncSqliteSaver` 而非同步 `SqliteSaver`：
项目图链路走 `graph.astream(...)`（异步），同步 `SqliteSaver.aget_tuple/aput`
直接抛 `NotImplementedError`（"The SqliteSaver does not support async methods"），
因此异步链路必须用异步 Saver。

⚠️ 事件循环绑定：`AsyncSqliteSaver` 在构造时即绑定当前 running loop，
其底层 `aiosqlite.Connection` 不可跨 loop 复用（跨 loop 使用会 `RuntimeError`）。
故 `persistent=True` 必须在运行中的事件循环内调用，调用方按 loop 缓存实例
（见 `api/chat_handler._get_graph_checkpointer`）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 项目根（agents/graph/checkpoint.py -> agents/graph -> agents -> 项目根）。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 既有默认库（与 config/database.py 的 DATABASE_URL 默认值一致）。
DEFAULT_DB_PATH = "data/smart_appointment.db"


def _strip_sqlite_url(value: str) -> str:
    for prefix in ("sqlite:///", "sqlite://", "sqlite:"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def resolve_db_path(db_path: Optional[str] = None) -> str:
    """解析落盘路径：显式入参 > 项目 `DATABASE_URL`（文件库） > 项目默认库。

    相对路径统一相对项目根解析，避免 CWD 不同时在第三方库目录下误建库。
    """
    candidate = db_path
    if not candidate:
        env = os.getenv("DATABASE_URL", "").strip()
        if env.startswith("sqlite"):
            parsed = _strip_sqlite_url(env)
            if parsed and parsed != ":memory:":
                candidate = parsed
    if not candidate:
        candidate = DEFAULT_DB_PATH

    path = Path(candidate)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def persistence_enabled(flag: Optional[bool] = None) -> bool:
    """`GRAPH_PERSISTENCE` 开关判定：显式入参优先，否则读 env（默认关）。"""
    if flag is not None:
        return bool(flag)
    return os.getenv("GRAPH_PERSISTENCE", "0").strip().lower() in ("1", "true", "yes", "on")


def build_checkpointer(db_path: Optional[str] = None, persistent: bool = True,
                       conn: Optional[Any] = None):
    """构造 checkpointer：`persistent=True` → AsyncSqliteSaver；否则 InMemorySaver。

    Args:
        db_path: 落盘库路径；缺省走 `resolve_db_path()`（既有 DB 配置 / 项目默认库）。
        persistent: True 用持久化 Saver；False 用进程内 `InMemorySaver`（默认语义不变）。
        conn: 可选注入的 `aiosqlite.Connection`，便于复用既有连接 / 测试。

    Raises:
        RuntimeError: `persistent=True` 但当前没有运行中的事件循环
            （`AsyncSqliteSaver` 绑定 loop，无法在纯同步上下文构造）。
    """
    if not persistent:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if conn is None:
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:  # pragma: no cover - 由调用方保证
            raise RuntimeError(
                "AsyncSqliteSaver 绑定事件循环，persistent=True 必须在运行中的 loop 内调用"
            ) from exc
        import aiosqlite

        conn = aiosqlite.connect(resolve_db_path(db_path))
    return AsyncSqliteSaver(conn)


async def close_checkpointer(checkpointer: Any) -> None:
    """关闭 AsyncSqliteSaver 底层连接；InMemorySaver / None 为无操作。"""
    conn = getattr(checkpointer, "conn", None)
    if conn is None:
        return
    try:
        await conn.close()
    except Exception:  # noqa: BLE001 - 关闭失败不应影响主流程
        logger.debug("checkpointer 关闭失败(已忽略)", exc_info=True)
