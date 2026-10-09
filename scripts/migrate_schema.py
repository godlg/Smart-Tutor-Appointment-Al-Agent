"""
一次性 schema migration 脚本（修复旧 SQLite 数据库的 schema 漂移）

背景：早期版本（按摩门店场景）的数据库 smart_appointment.db 已存在，
ORM 重构为家教场景后：
  - user_behaviors 表缺 teacher_id 列（旧版是 technician_id）
  - 残留 technicians / technician_schedules 两张旧表
  - ORM 的 Base.metadata.create_all() 不会修改已存在的表/列

SessionManager 启动时会自动调用 db.base.schema_migrator.run_migration，
所以正常情况下不需要手工跑这个脚本；仅在调试/排错时显式执行。

本脚本打印详细日志（含 PRAGMA 表结构），便于确认迁移结果。

用法：
  python scripts/migrate_schema.py                # 自动定位 data/smart_appointment.db
  python scripts/migrate_schema.py /path/to.db    # 自定义 db 路径

退出码：
  0 = 成功（无论是否有变更）
  非 0 = 失败
"""
import sqlite3
import sys
from pathlib import Path


def default_db_path():
    here = Path(__file__).resolve().parent
    return here.parent / "data" / "smart_appointment.db"


def migrate(db_path: Path) -> int:
    if not db_path.exists():
        print(f"[skip] {db_path} 不存在，无需 migrate（首次启动会自动 create_all）")
        return 0

    # 把项目根加进 sys.path 才能 import db.base.schema_migrator
    proj_root = db_path.parent.parent
    sys.path.insert(0, str(proj_root))
    from db.base.schema_migrator import (
        _get_columns, _table_exists, run_migration,
    )

    if not run_migration(str(db_path)):
        return 1

    # 打印最终 schema
    conn = sqlite3.connect(str(db_path))
    try:
        print("\n=== 当前 user_behaviors schema ===")
        for r in conn.execute("PRAGMA table_info(user_behaviors)").fetchall():
            print(f"  {r}")
        print("\n=== 残留旧表检查 ===")
        for old in ("technicians", "technician_schedules"):
            print(f"  {old}: {'存在' if _table_exists(conn, old) else '已删 ✓'}")
    finally:
        conn.close()
    return 0


def main():
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = default_db_path()

    print(f"DB: {db_path}")
    sys.exit(migrate(db_path))


if __name__ == "__main__":
    main()