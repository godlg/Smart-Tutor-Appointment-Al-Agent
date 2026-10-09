from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from ..models import Base


class SessionManager:
    """
    数据库会话管理器

    职责:
    1. 管理数据库连接和会话
    2. 提供统一的会话上下文管理
    3. 处理事务和异常回滚
    """

    def __init__(self, db_path='sqlite:///data/smart_appointment.db'):
        """
        初始化会话管理器

        Args:
            db_path: 数据库连接路径
        """
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        # create_all 不会修改已存在表的列（schema 漂移）。
        # 对 SQLite 文件型数据库，跑一次轻量 migration 把旧 schema 拉到新 schema。
        # 文件型 SQLite 必须迁移成功；缺唯一约束不能继续声称记忆可靠。
        database = self.engine.url.database
        if self.engine.dialect.name == "sqlite" and database not in (None, "", ":memory:"):
            from .schema_migrator import run_migration
            if not run_migration(database):
                self.engine.dispose()
                raise RuntimeError("memory schema 迁移失败，请人工核查旧库；不会自动删除数据")
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    @contextmanager
    def session_scope(self):
        """
        提供会话上下文管理
        
        自动处理：
        - 会话创建和关闭
        - 事务提交和回滚
        - 异常处理
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """关闭会话管理器"""
        self.Session.remove()
        self.engine.dispose()
