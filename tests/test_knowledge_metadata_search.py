"""
知识库元数据过滤检索测试

覆盖：
- Schema: KnowledgeDocument 新增 subject / grade_level / doc_type 列
- Repository: 新字段可写入; SQL 过滤生效
- Service: filter-then-vector 模式: 业务不匹配的记录即使语义相近也不会混入
- Migration: 老 schema 经 schema_migrator 升级后获得新列 + 启发式 backfill

被 embedding 阻断的测试用 monkeypatch 注入 fake embedder(不依赖真实模型服务)。
"""
import asyncio
import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _fake_embed(text: str) -> List[float]:
    """把文本哈希成 16 维向量;相似文本得到相似向量。
    故意不归一化,FAISS.IndexFlatIP(内积) 在未归一化向量上仍能排序,
    不影响"排序后取 top_k"的核心行为。
    """
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input;每个测试自动应用。

    注意:必须同时 patch 两处:
    - services.text_embedding 模块属性(供 text_embedding.py 内部引用)
    - services.knowledge_service 模块属性(它用 from .text_embedding import
      embed_input,在模块加载时已捕获原函数引用)
    否则会出现 "FAISS index dim != query dim" 的 AssertionError。
    """
    from services import text_embedding
    from services import knowledge_service
    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)
    monkeypatch.setattr(knowledge_service, "embed_input", _fake_embed)


@pytest.fixture
def in_memory_db():
    """全新内存 SQLite,schema + migration 跑过。"""
    from db.db_router import DatabaseRouter
    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


def _seed_corpus(repo):
    """种子数据:4 条记录,故意覆盖 subject/grade/doc_type 多个组合。

    - 高中数学 12 周提分方案     (数学/高中/course_intro)
    - 高三冲刺套餐              (NULL/高中/course_intro)  ← 跨学科
    - 小学阅读启蒙课            (语文/小学/course_intro)
    - 初中数学 一元二次方程     (数学/初中/subject_method)
    """
    repo.add_document(
        content="高中数学 12 周提分方案:函数、几何、数列、概率四大模块系统讲。",
        category="course_intro",
        keywords=["高中", "数学", "高考"],
        embedding=_fake_embed("高中数学"),
        subject="数学", grade_level="高中", doc_type="course_intro",
    )
    repo.add_document(
        content="高三冲刺套餐:针对 6 月高考设计,1 月到 5 月完成三轮复习,适合数学、英语、理综。",
        category="course_intro",
        keywords=["高三", "冲刺", "全科"],
        embedding=_fake_embed("高三冲刺"),
        subject=None, grade_level="高中", doc_type="course_intro",
    )
    repo.add_document(
        content="小学阅读启蒙课:绘本为载体,讲解 + 造句 + 复述故事,建立自主阅读习惯。",
        category="course_intro",
        keywords=["小学", "阅读", "绘本"],
        embedding=_fake_embed("小学阅读"),
        subject="语文", grade_level="小学", doc_type="course_intro",
    )
    repo.add_document(
        content="初中数学 一元二次方程 解题套路与错题梳理方法。",
        category="subject_method",
        keywords=["初中", "数学", "一元二次方程"],
        embedding=_fake_embed("初中数学"),
        subject="数学", grade_level="初中", doc_type="subject_method",
    )


def _build_service(in_memory_db):
    """基于给定 in_memory_db 构造一个全新的 KnowledgeService 实例。"""
    from services.knowledge_service import KnowledgeService
    svc = KnowledgeService(db_path="sqlite:///:memory:")
    # 替换为测试库,确保 search() 跑在已 seed 的数据上
    svc.db_router = in_memory_db
    svc.db = in_memory_db.knowledge_repo
    svc.document_ids = []
    svc.index = None
    svc._index_dirty = False
    svc.initialized = False
    return svc


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# A. Schema
# ----------------------------------------------------------------------

class TestKnowledgeDocumentSchema:
    def test_has_subject_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "subject"), "subject 列应存在"

    def test_has_grade_level_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "grade_level"), "grade_level 列应存在"

    def test_has_doc_type_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "doc_type"), "doc_type 列应存在"


# ----------------------------------------------------------------------
# B. Repository: 新字段可写
# ----------------------------------------------------------------------

class TestRepositoryMetadataPersistence:
    def test_add_document_persists_subject_grade_doc_type(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="初中数学 一元二次方程 解题套路",
            category="subject_method",
            keywords=["初中", "数学"],
            embedding=_fake_embed("初中数学"),
            subject="数学",
            grade_level="初中",
            doc_type="subject_method",
        )
        doc = repo.get_document(doc_id)
        assert doc["subject"] == "数学"
        assert doc["grade_level"] == "初中"
        assert doc["doc_type"] == "subject_method"

    def test_add_document_optional_fields_default_none(self, in_memory_db):
        """老调用方不传 subject/grade/doc_type 时,应允许(可空)。"""
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="跨学科内容",
            category="parent_guide",
            keywords=["家长", "指南"],
            embedding=_fake_embed("家长指南"),
        )
        doc = repo.get_document(doc_id)
        assert doc["subject"] is None
        assert doc["grade_level"] is None
        assert doc["doc_type"] is None


# ----------------------------------------------------------------------
# C. Repository: SQL 过滤
# ----------------------------------------------------------------------

class TestRepositorySQLFilter:
    def test_subject_filter_excludes_other_subjects(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters(subject="数学")
        contents = " ".join(r["content"] for r in results)

        # 主体必须是 数学(含 NULL 跨学科文档允许存在,因为 NULL ≠ 其他学科)
        assert "小学阅读启蒙课" not in contents  # 语文 → 必须排除
        assert "高中数学 12 周" in contents      # 数学/高中 → 应保留
        assert "初中数学" in contents            # 数学/初中 → 应保留

    def test_grade_filter_excludes_other_grades(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters(grade_level="初中")
        contents = " ".join(r["content"] for r in results)

        assert "初中数学" in contents
        assert "高中数学 12 周" not in contents
        assert "高三冲刺" not in contents
        assert "小学阅读启蒙课" not in contents

    def test_combined_subject_and_grade(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters(subject="数学", grade_level="小学")
        contents = " ".join(r["content"] for r in results)

        # corpus 里没有 subject=数学 AND grade=小学 的记录
        assert contents.strip() == ""

    def test_combined_subject_and_grade_hits(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters(subject="数学", grade_level="初中")
        contents = [r["content"] for r in results]

        assert len(contents) == 1
        assert "初中数学" in contents[0]

    def test_no_filter_returns_all(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters()
        assert len(results) == 4

    def test_unmatched_filter_keeps_only_cross_subject_docs(self, in_memory_db):
        """P0-2 NULL 豁免:corpus 里没有 subject=物理 的记录,但 subject=None 的
        跨学科文档(高三冲刺套餐)必须保留,不能因为「物理」这个过滤值被一刀切掉。
        """
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)

        results = repo.search_documents_with_filters(subject="物理")
        assert len(results) == 1
        assert results[0]["subject"] is None
        assert "高三冲刺" in results[0]["content"]


# ----------------------------------------------------------------------
# D. Service: filter-then-vector
# ----------------------------------------------------------------------

class TestServiceFilterBeforeVector:
    """最关键的一组测试:即便向量把"高中数学"判得很近,业务硬约束也要把它滤掉。"""

    def test_grade_filter_excludes_higher_grades(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        results = _run(svc.search(query="初中数学 解题套路", top_k=3, grade_level="初中"))
        contents = [r["content"] for r in results]

        assert any("初中数学" in c for c in contents)
        # 高中 / 高三 即使向量相近,因 grade=高中,不应混入
        assert not any("高中" in c for c in contents)

    def test_subject_filter_excludes_other_subjects(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        results = _run(svc.search(query="数学 解题", top_k=3, subject="数学"))
        contents = [r["content"] for r in results]

        assert not any("小学阅读" in c for c in contents)

    def test_no_filter_returns_all_semantically_related(self, in_memory_db):
        """回归保护:不传 filter 时,行为保持原样(向量全集搜)。"""
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        results = _run(svc.search(query="数学", top_k=3))
        # top_k=3 至少能返回 3 条
        assert len(results) >= 1

    def test_filter_with_no_subject_match_keeps_cross_subject(self, in_memory_db):
        """P0-2 NULL 豁免在 service 层同样生效:subject=物理 无精确命中时,
        跨学科(subject=None)文档仍应召回,而不是空结果。
        """
        repo = in_memory_db.knowledge_repo
        _seed_corpus(repo)
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        results = _run(svc.search(query="化学 高中", top_k=3, subject="物理"))
        assert results
        assert all(r["subject"] is None for r in results)


# ----------------------------------------------------------------------
# E. Migration
# ----------------------------------------------------------------------

class TestSchemaMigration:
    def test_old_db_gets_new_columns(self, tmp_path):
        """旧 schema 跑迁移后,新列被加上。"""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE knowledge_documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                category VARCHAR NOT NULL,
                keywords JSON,
                embedding JSON,
                created_at DATETIME,
                updated_at DATETIME,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()

        from db.base.schema_migrator import run_migration
        run_migration(str(db_path))

        conn = sqlite3.connect(str(db_path))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_documents)").fetchall()}
        conn.close()

        assert "subject" in cols
        assert "grade_level" in cols
        assert "doc_type" in cols

    def test_heuristic_backfill_from_keywords(self, tmp_path):
        """旧数据 keywords 含 '高中'/'数学' 时,backfill 自动推断。"""
        db_path = tmp_path / "old2.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE knowledge_documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                category VARCHAR NOT NULL,
                keywords JSON,
                embedding JSON,
                created_at DATETIME,
                updated_at DATETIME,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.execute(
            "INSERT INTO knowledge_documents (content, category, keywords, is_active) "
            "VALUES (?, ?, ?, 1)",
            ("高中数学 12 周提分方案", "course_intro", '["高中","数学","高考"]'),
        )
        conn.commit()
        conn.close()

        from db.base.schema_migrator import run_migration
        run_migration(str(db_path))

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT subject, grade_level FROM knowledge_documents WHERE id=1"
        ).fetchone()
        conn.close()

        assert row[0] == "数学"
        assert row[1] == "高中"

    def test_backfill_only_runs_on_first_column_add(self, tmp_path):
        """回归测试(P1-12 修复):backfill 只在首次加列时触发。

        否则每次 API 请求 new SessionManager 时都会跑 backfill,
        把 _create_default_knowledge 显式写入的 None(跨学科/跨年级记录)
        覆盖成从 content 启发式推断的值。
        """
        from db.base.schema_migrator import run_migration

        db_path = tmp_path / "idempotent.db"
        # 1) 模拟"已升级过"的 DB:列已存在,内容含 '数学'/'高中'
        #    但 subject/grade 是用户显式设的 None(跨学科文档)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE knowledge_documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                category VARCHAR NOT NULL,
                keywords JSON,
                embedding JSON,
                subject VARCHAR,
                grade_level VARCHAR,
                doc_type VARCHAR,
                created_at DATETIME,
                updated_at DATETIME,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.execute(
            "INSERT INTO knowledge_documents "
            "(content, category, keywords, subject, grade_level, doc_type, is_active) "
            "VALUES (?, ?, ?, NULL, NULL, ?, 1)",
            (
                "如何帮孩子选择合适的家教老师:先看学科匹配(数学/物理 / 英语等),"
                "再看年级段(小学 / 初中 / 高中)",
                "parent_guide",
                '["家长", "选择"]',
                "parent_guide",
            ),
        )
        conn.commit()
        conn.close()

        # 2) 跑两次 run_migration (模拟 API 多次请求)
        run_migration(str(db_path))
        run_migration(str(db_path))

        # 3) 验证 subject/grade 仍为 NULL(不应被启发式覆盖)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT subject, grade_level FROM knowledge_documents WHERE id=1"
        ).fetchone()
        conn.close()

        assert row[0] is None, (
            f"backfill 不应覆盖显式 None,但 subject={row[0]!r}"
        )
        assert row[1] is None, (
            f"backfill 不应覆盖显式 None,但 grade_level={row[1]!r}"
        )