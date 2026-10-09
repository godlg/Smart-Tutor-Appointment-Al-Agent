"""旧库冲突时失败封闭，不删业务数据或伪称唯一索引已建立。"""

import sqlite3

import pytest


@pytest.mark.parametrize("table, columns, rows", [
    ("session_messages", "session_id TEXT, user_id TEXT, seq INTEGER",
     "('s','owner',1),('s','owner',1)"),
    ("session_summaries", "session_id TEXT, user_id TEXT, start_seq INTEGER, end_seq INTEGER",
     "('s','owner',1,2),('s','owner',1,2)"),
    ("chat_sessions", "session_id TEXT, user_id TEXT",
     "('s','alice'),('s','bob')"),
])
def test_critical_uniqueness_conflicts_block_startup_without_deleting_rows(tmp_path, table, columns, rows):
    from db.base.schema_migrator import run_migration
    from db.base.session_manager import SessionManager

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, {columns})")
        fields = ",".join(part.strip().split()[0] for part in columns.split(","))
        connection.execute(f"INSERT INTO {table} ({fields}) VALUES {rows}")
    assert run_migration(str(path)) is False
    with pytest.raises(RuntimeError, match="memory.*schema|记忆.*迁移"):
        SessionManager(f"sqlite:///{path}")
    with sqlite3.connect(path) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 2


def test_partial_unique_index_cannot_impersonate_full_message_constraint(tmp_path):
    from db.base.schema_migrator import run_migration
    path = tmp_path / "partial-index.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE session_messages (id INTEGER PRIMARY KEY, session_id TEXT,
                user_id TEXT, seq INTEGER, status TEXT);
            CREATE UNIQUE INDEX ux_session_messages_session_seq
                ON session_messages(session_id,seq) WHERE status='complete';
            INSERT INTO session_messages VALUES (1,'s','owner',1,'partial');
            INSERT INTO session_messages VALUES (2,'s','owner',1,'partial');
        """)
    assert run_migration(str(path)) is False
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0] == 2
