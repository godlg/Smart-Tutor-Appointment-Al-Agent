"""
轻量级 SQLite schema migration（被 SessionManager 启动时调用）

设计原则:
- 幂等（重跑结果一致）
- 返回迁移成功状态；关键记忆唯一约束失败时启动层拒绝就绪，不清理旧数据
- 只做加列、建表和建索引;保留旧业务表与数据
- 与 scripts/migrate_schema.py 共用同一份逻辑（命令行入口调用 run_migration）

要做的事:
1. user_behaviors: 加 teacher_id 列（如果缺失），把旧 technician_id 数据复制过来
2. user_recommendations: 兜底加列 (teacher_id / is_sent / sent_at / created_at),
   旧 DB 在 user_recommendations 上缺这些列会让推荐写入和拉取直接 500
3. knowledge_documents (P1-12): 加 subject / grade_level / doc_type 列(全部 nullable),
   对老记录做启发式 backfill: 从 keywords 字段推断
4. 保留旧表 technicians / technician_schedules;迁移不删除历史业务数据
"""
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


def _get_columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn, table):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


# 启发式 backfill 用映射表:
# 关键字 → 学科/年级/类型。
# 故意保持保守:宁可不填,也不填错;NULL 比错值更安全(过滤时不会误判)。
_KEYWORD_TO_SUBJECT = {
    "数学": "数学", "语文": "语文", "英语": "英语", "物理": "物理",
    "化学": "化学", "生物": "生物", "历史": "历史", "地理": "地理", "政治": "政治",
}

_KEYWORD_TO_GRADE = {
    "高三": "高中", "高中": "高中",
    "初三": "初中", "初中": "初中", "初二": "初中", "初一": "初中", "中考": "初中",
    "小学": "小学",
}


def _backfill_knowledge_metadata(conn) -> None:
    """对老记录的 subject / grade_level 做启发式回填。

    优先级:
    1. 解析 keywords JSON 找学科/年级关键字(置信度最高)
    2. 回填 content 里出现的高频关键字(置信度次之)

    同一字段被多次匹配时,保留先匹配到的值(SQLite UPDATE 的默认行为)。
    """
    cols = _get_columns(conn, "knowledge_documents")
    if not {"subject", "grade_level"}.issubset(cols):
        return  # 列都没加上,backfill 无意义

    rows = conn.execute(
        "SELECT id, content, keywords FROM knowledge_documents"
    ).fetchall()
    for row_id, content, keywords_json in rows:
        # keywords 可能是 list(JSON) / 旧文本 / None
        kw_list: list[str] = []
        if keywords_json:
            try:
                parsed = json.loads(keywords_json)
                if isinstance(parsed, list):
                    kw_list = [str(k) for k in parsed]
                elif isinstance(parsed, str):
                    kw_list = [parsed]
            except (json.JSONDecodeError, TypeError):
                kw_list = [keywords_json]
        haystack = " ".join(kw_list + [content or ""])

        # 学科推断
        subject = None
        for kw, value in _KEYWORD_TO_SUBJECT.items():
            if kw in haystack:
                subject = value
                break

        # 年级推断(优先级:具体年级 > 段)
        grade = None
        # 先匹配更具体的("高三" 优先于 "高中")
        for kw in ("高三", "初三", "初二", "初一", "中考"):
            if kw in haystack:
                grade = _KEYWORD_TO_GRADE[kw]
                break
        if grade is None:
            for kw in ("高中", "初中", "小学"):
                if kw in haystack:
                    grade = _KEYWORD_TO_GRADE[kw]
                    break

        updates = []
        params = []
        if subject:
            updates.append("subject = ?")
            params.append(subject)
        if grade:
            updates.append("grade_level = ?")
            params.append(grade)
        if updates:
            params.append(row_id)
            conn.execute(
                f"UPDATE knowledge_documents SET {', '.join(updates)} WHERE id = ?",
                params,
            )


def run_migration(db_path: str) -> bool:
    """
    执行 schema 迁移。返回 True 表示完成（无论有无变更），False/异常表示跳过。

    注意:此函数每次新建 SessionManager 时都会跑(P1-12 修复):
    - knowledge_documents 的启发式 backfill **只在首次加列的那次** 跑,
      否则会覆盖用户显式设置的 None 值(跨学科/跨年级记录会被错填)。
    """
    if not db_path or db_path == ":memory:":
        logger.info("schema_migrator: in-memory db, skip")
        return True

    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "user_behaviors"):
            logger.info("schema_migrator: user_behaviors 不存在,跳过 user_behaviors 迁移")
        else:
            # 1) teacher_id 列
            cols = _get_columns(conn, "user_behaviors")
            if "teacher_id" not in cols:
                conn.execute("ALTER TABLE user_behaviors ADD COLUMN teacher_id INTEGER")
                logger.info("schema_migrator: user_behaviors.teacher_id 列已添加")

            # 2) technician_id -> teacher_id 数据迁移
            cols = _get_columns(conn, "user_behaviors")
            if "technician_id" in cols:
                cur = conn.execute(
                    "UPDATE user_behaviors "
                    "SET teacher_id = technician_id "
                    "WHERE teacher_id IS NULL AND technician_id IS NOT NULL"
                )
                logger.info(
                    f"schema_migrator: technician_id->teacher_id 复制 {cur.rowcount} 行"
                )

        # 2.5) user_recommendations 兜底:旧 DB 可能缺列
        if _table_exists(conn, "user_recommendations"):
            rec_cols = _get_columns(conn, "user_recommendations")
            required = {
                "teacher_id": "INTEGER",
                "is_sent": "INTEGER DEFAULT 0",
                "created_at": "DATETIME",
                "sent_at": "DATETIME",
            }
            for col, decl in required.items():
                if col not in rec_cols:
                    try:
                        conn.execute(
                            f"ALTER TABLE user_recommendations ADD COLUMN {col} {decl}"
                        )
                        logger.info(
                            f"schema_migrator: user_recommendations.{col} 列已添加"
                        )
                    except Exception as e:
                        logger.warning(
                            f"schema_migrator: 加 user_recommendations.{col} 失败: {e}"
                        )
        else:
            logger.info(
                "schema_migrator: user_recommendations 不存在,稍后由 SQLAlchemy create_all 创建"
            )

        # 2.7) knowledge_documents (P1-12): 加 3 个元数据列 + 仅首次触发 backfill
        columns_added_this_run = False
        if _table_exists(conn, "knowledge_documents"):
            kd_cols = _get_columns(conn, "knowledge_documents")
            new_cols = {
                "subject": "VARCHAR",
                "grade_level": "VARCHAR",
                "doc_type": "VARCHAR",
            }
            for col, decl in new_cols.items():
                if col not in kd_cols:
                    try:
                        conn.execute(
                            f"ALTER TABLE knowledge_documents ADD COLUMN {col} {decl}"
                        )
                        logger.info(
                            f"schema_migrator: knowledge_documents.{col} 列已添加"
                        )
                        columns_added_this_run = True
                    except Exception as e:
                        logger.warning(
                            f"schema_migrator: 加 knowledge_documents.{col} 失败: {e}"
                        )
            # 关键:backfill 只在本次运行真的加了列时才触发
            # 否则每次 API 请求都 new SessionManager 时,backfill 会覆盖
            # _create_default_knowledge 显式写入的 None(跨学科/跨年级记录)
            if columns_added_this_run:
                try:
                    _backfill_knowledge_metadata(conn)
                    logger.info(
                        "schema_migrator: knowledge_documents 启发式 backfill 完成(首次加列)"
                    )
                except Exception as e:
                    logger.warning(
                        f"schema_migrator: knowledge_documents backfill 失败: {e}"
                    )
            else:
                logger.debug(
                    "schema_migrator: knowledge_documents 列已存在,跳过 backfill"
                )

        # 2.8) knowledge_documents (P1-13): 上传溯源列。
        # 不做 backfill —— 老数据无源文件信息,NULL 是正确值。
        columns_added_this_run_upload = False
        if _table_exists(conn, "knowledge_documents"):
            kd_cols_upload = _get_columns(conn, "knowledge_documents")
            new_upload_cols = {
                "source_file": "VARCHAR",
                "chunk_index": "INTEGER",
                "source_hash": "VARCHAR",
                "chunk_count": "INTEGER",
            }
            for col, decl in new_upload_cols.items():
                if col not in kd_cols_upload:
                    try:
                        conn.execute(
                            f"ALTER TABLE knowledge_documents ADD COLUMN {col} {decl}"
                        )
                        logger.info(
                            f"schema_migrator: knowledge_documents.{col} 列已添加 (P1-13)"
                        )
                        columns_added_this_run_upload = True
                    except Exception as e:
                        logger.warning(
                            f"schema_migrator: 加 knowledge_documents.{col} 失败: {e}"
                        )

        # 3) AutoDream: 后台任务三张表(checkpoints / locks / change_logs)。
        # 用 CREATE TABLE IF NOT EXISTS 幂等建表,兼容"旧库已存在、ORM create_all 不补新表"
        # 的 schema 漂移场景;表结构以 db/models.py 的 ORM 定义为准。
        autodream_tables = {
            "autodream_checkpoints": (
                "CREATE TABLE IF NOT EXISTS autodream_checkpoints ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL, "
                "last_event_id INTEGER DEFAULT 0, "
                "last_run_at DATETIME, "
                "total_replays INTEGER DEFAULT 0)"
            ),
            "autodream_locks": (
                "CREATE TABLE IF NOT EXISTS autodream_locks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL, "
                "locked_at DATETIME, "
                "lock_token VARCHAR, "
                "expires_at DATETIME)"
            ),
            "autodream_change_logs": (
                "CREATE TABLE IF NOT EXISTS autodream_change_logs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL, "
                "event_id INTEGER NOT NULL, "
                "action VARCHAR NOT NULL, "
                "preference_type VARCHAR NOT NULL, "
                "preference_value VARCHAR NOT NULL, "
                "detail TEXT, "
                "created_at DATETIME)"
            ),
        }
        for table, ddl in autodream_tables.items():
            if not _table_exists(conn, table):
                conn.execute(ddl)
                logger.info(f"schema_migrator: 建 AutoDream 表 {table}")

        # 3.5) long_term_memories (分层记忆增强): 内容型长期记忆表。
        # 由 SQLAlchemy create_all 在 SessionManager.__init__ 里也会建,这里幂等兜底
        # "老库已存在、create_all 不补新表" 的场景。
        if not _table_exists(conn, "long_term_memories"):
            conn.execute(
                "CREATE TABLE IF NOT EXISTS long_term_memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL, "
                "memory_type VARCHAR NOT NULL, "
                "content TEXT NOT NULL, "
                "embedding TEXT, "
                "importance FLOAT DEFAULT 0.5, "
                "session_id VARCHAR, "
                "created_at DATETIME, "
                "last_accessed_at DATETIME)"
            )
            logger.info("schema_migrator: 建长期记忆表 long_term_memories")

        # 3.55) 记忆可靠性 Phase 1: 匿名身份 / 会话归属 / 顺序消息 / 滚动摘要 / 检查点。
        # 幂等 CREATE TABLE IF NOT EXISTS,兼容"旧库已存在、create_all 不补新表"的漂移
        # 场景;表结构以 db/models.py 的 ORM 定义为准。不做 backfill —— 老记录无身份/
        # 会话信息,NULL/不存在是正确值,未知归属不得向新匿名身份暴露 default_user 数据。
        memory_reliability_tables = {
            "identities": (
                "CREATE TABLE IF NOT EXISTS identities ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL UNIQUE, "
                "created_at DATETIME)"
            ),
            "identity_students": (
                "CREATE TABLE IF NOT EXISTS identity_students ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id VARCHAR NOT NULL, "
                "student_id VARCHAR NOT NULL UNIQUE, "
                "created_at DATETIME)"
            ),
            "chat_sessions": (
                "CREATE TABLE IF NOT EXISTS chat_sessions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id VARCHAR NOT NULL UNIQUE, "
                "user_id VARCHAR NOT NULL, "
                "created_at DATETIME, "
                "updated_at DATETIME)"
            ),
            "session_messages": (
                "CREATE TABLE IF NOT EXISTS session_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id VARCHAR NOT NULL, "
                "user_id VARCHAR NOT NULL, "
                "seq INTEGER NOT NULL, "
                "role VARCHAR NOT NULL, "
                "content TEXT NOT NULL, "
                "status VARCHAR DEFAULT 'complete', "
                "created_at DATETIME)"
            ),
            "session_summaries": (
                "CREATE TABLE IF NOT EXISTS session_summaries ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id VARCHAR NOT NULL, "
                "user_id VARCHAR NOT NULL, "
                "start_seq INTEGER NOT NULL, "
                "end_seq INTEGER NOT NULL, "
                "content TEXT NOT NULL, "
                "created_at DATETIME)"
            ),
            "session_checkpoints": (
                "CREATE TABLE IF NOT EXISTS session_checkpoints ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id VARCHAR NOT NULL UNIQUE, "
                "user_id VARCHAR NOT NULL, "
                "state VARCHAR, "
                "payload TEXT, "
                "pending_ops TEXT, "
                "updated_at DATETIME)"
            ),
        }
        for table, ddl in memory_reliability_tables.items():
            if not _table_exists(conn, table):
                conn.execute(ddl)
                logger.info(f"schema_migrator: 建记忆可靠性表 {table}")

        # Additive only: legacy student ownership is unknown, so retain NULL.
        for table in ("chat_sessions", "session_messages", "session_summaries",
                      "long_term_memories"):
            if "student_id" not in _get_columns(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN student_id VARCHAR")
        # 半成品旧库也可能只有部分列。未知历史保留 NULL，不伪造内容或 owner。
        compatibility_columns = {
            "chat_sessions": {"created_at": "DATETIME", "updated_at": "DATETIME"},
            "session_messages": {"role": "VARCHAR", "content": "TEXT",
                                 "status": "VARCHAR", "created_at": "DATETIME"},
            "session_summaries": {"content": "TEXT", "created_at": "DATETIME"},
            "session_checkpoints": {"state": "VARCHAR", "payload": "TEXT",
                                    "pending_ops": "TEXT", "updated_at": "DATETIME"},
            "long_term_memories": {
                "session_id": "VARCHAR", "memory_type": "VARCHAR", "embedding": "JSON",
                "importance": "FLOAT", "created_at": "DATETIME", "last_accessed_at": "DATETIME",
                "start_seq": "INTEGER", "end_seq": "INTEGER",
            },
        }
        for table, columns in compatibility_columns.items():
            existing = _get_columns(conn, table)
            for column, declaration in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

        # 唯一键 / 查询索引(session_messages 的 (session_id, seq) 唯一键保障顺序幂等)。
        _index_ddls = [
            "CREATE INDEX IF NOT EXISTS ix_identities_user_id ON identities(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_identity_students_user_id ON identity_students(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_identity_students_student_id ON identity_students(student_id)",
            "CREATE INDEX IF NOT EXISTS ix_chat_sessions_session_id ON chat_sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions(user_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_sessions_session_id ON chat_sessions(session_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_session_checkpoints_session_id ON session_checkpoints(session_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_session_messages_session_seq "
            "ON session_messages(session_id, seq)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_session_summaries_span "
            "ON session_summaries(session_id, start_seq, end_seq)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_long_term_memories_span "
            "ON long_term_memories(session_id, start_seq, end_seq)",
        ]
        for ddl in _index_ddls:
            try:
                conn.execute(ddl)
            except sqlite3.Error as e:
                if "CREATE UNIQUE INDEX" in ddl:
                    raise
                logger.warning("schema_migrator: 可选查询索引未建立: %s", type(e).__name__)

        required_keys = {
            "chat_sessions": ("session_id",),
            "session_checkpoints": ("session_id",),
            "session_messages": ("session_id", "seq"),
            "session_summaries": ("session_id", "start_seq", "end_seq"),
            "long_term_memories": ("session_id", "start_seq", "end_seq"),
        }
        for table, key in required_keys.items():
            unique_keys = {
                tuple(row[2] for row in conn.execute(f'PRAGMA index_info("{index[1]}")'))
                for index in conn.execute(f"PRAGMA index_list({table})")
                if index[2] and not index[4]
            }
            if key not in unique_keys:
                raise sqlite3.IntegrityError("记忆核心唯一约束未建立")

        # 3.6) human_handoffs (人工接管 P1): 补 transcript / agent_id 列(均 nullable)。
        # 不做 backfill —— 老记录无快照/认领信息,NULL 是正确值。
        # 新表 human_handoff_messages 由 create_all 自动创建,不在此写迁移。
        if _table_exists(conn, "human_handoffs"):
            hh_cols = _get_columns(conn, "human_handoffs")
            hh_new_cols = {
                "transcript": "TEXT",
                "agent_id": "VARCHAR",
            }
            for col, decl in hh_new_cols.items():
                if col not in hh_cols:
                    try:
                        conn.execute(
                            f"ALTER TABLE human_handoffs ADD COLUMN {col} {decl}"
                        )
                        logger.info(
                            f"schema_migrator: human_handoffs.{col} 列已添加"
                        )
                    except Exception as e:
                        logger.warning(
                            f"schema_migrator: 加 human_handoffs.{col} 失败: {e}"
                        )
        else:
            logger.info(
                "schema_migrator: human_handoffs 不存在,稍后由 create_all 创建"
            )

        # Preserve legacy business tables and records. Retirement is not deletion.

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.warning("schema_migrator: 迁移失败，保留旧数据: %s", type(e).__name__)
        return False
    finally:
        conn.close()