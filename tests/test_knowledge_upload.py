"""
知识库上传功能测试 (P1-13: PDF / Markdown 批量入库)

覆盖:
- Schema: KnowledgeDocument 新增 source_file / chunk_index / source_hash / chunk_count 列
- Migration: schema_migrator 给老 schema 加列,老行保持 NULL,无 backfill
- Repository: 新字段可写入; find_chunks_by_source_hash 按 chunk_index ASC 排序
- DocumentParser: detect_kind / parse_pdf / parse_markdown / parse 入口
- TextChunker: split 行为 (短文本单块 / 长文本多块 / chunk_index 连续 / 空文本报错)
- Service: add_uploaded_document 走完 parse → chunk → embed → 写库 → FAISS 增量
- API: POST /api/knowledge/upload multipart/form-data 端到端

embedding 用 monkeypatch fake (16-dim SHA256) 替身,沿用 tests/test_knowledge_metadata_search.py 的双命名空间套路。
"""
import asyncio
import hashlib
import io
import sqlite3
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


# ----------------------------------------------------------------------
# Helpers & fixtures
# ----------------------------------------------------------------------

def _fake_embed(text: str) -> List[float]:
    """与 test_knowledge_metadata_search 保持一致的 16 维假向量"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input;同时 patch 两处 (text_embedding 与 knowledge_service)
    避免 from .text_embedding import embed_input 捕获原引用。"""
    from services import text_embedding
    from services import knowledge_service
    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)
    monkeypatch.setattr(knowledge_service, "embed_input", _fake_embed)


@pytest.fixture
def in_memory_db():
    from db.db_router import DatabaseRouter
    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


def _run(coro):
    return asyncio.run(coro)


def _build_service(in_memory_db):
    """在测试用 in-memory DB 上挂一个干净 KnowledgeService 实例。"""
    from services.knowledge_service import KnowledgeService
    svc = KnowledgeService(db_path="sqlite:///:memory:")
    svc.db_router = in_memory_db
    svc.db = in_memory_db.knowledge_repo
    svc.document_ids = []
    svc.index = None
    svc._index_dirty = False
    svc.initialized = False
    return svc


# ----------------------------------------------------------------------
# A. Schema: 4 个新列必须存在
# ----------------------------------------------------------------------

class TestKnowledgeDocumentSchemaUpload:
    def test_has_source_file_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "source_file"), \
            "source_file 列应存在 (upload 溯源)"

    def test_has_chunk_index_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "chunk_index"), \
            "chunk_index 列应存在"

    def test_has_source_hash_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "source_hash"), \
            "source_hash 列应存在 (去重键)"

    def test_has_chunk_count_column(self):
        from db.models import KnowledgeDocument
        assert hasattr(KnowledgeDocument, "chunk_count"), \
            "chunk_count 列应存在"


# ----------------------------------------------------------------------
# B. Migration: 老 DB 加 4 列 + 不抛异常 + 老行 NULL 不被覆盖
# ----------------------------------------------------------------------

class TestSchemaMigrationUpload:
    """P1-13: schema_migrator 给老 schema 加新列;
    不做启发式 backfill (无法从内容反推文件名)。
    """

    OLD_SCHEMA_SQL = """
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
    """

    def test_old_db_gets_upload_columns(self, tmp_path):
        """老 schema 跑迁移后,4 个新列全部加上。"""
        db_path = tmp_path / "old_with_subject_grade.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(self.OLD_SCHEMA_SQL)
        conn.commit()
        conn.close()

        from db.base.schema_migrator import run_migration
        run_migration(str(db_path))

        conn = sqlite3.connect(str(db_path))
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(knowledge_documents)"
        ).fetchall()}
        conn.close()

        assert "source_file" in cols
        assert "chunk_index" in cols
        assert "source_hash" in cols
        assert "chunk_count" in cols

    def test_idempotent_migration_runs_twice(self, tmp_path):
        """迁移跑两次不抛异常 (与 P1-12 同款保证)。"""
        db_path = tmp_path / "idempotent_upload.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(self.OLD_SCHEMA_SQL)
        conn.commit()
        conn.close()

        from db.base.schema_migrator import run_migration
        run_migration(str(db_path))
        # 第二次不应抛异常
        run_migration(str(db_path))

    def test_old_rows_have_null_provenance(self, tmp_path):
        """老数据迁移后 4 列必须为 NULL,不做 backfill。"""
        db_path = tmp_path / "null_provenance.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(self.OLD_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO knowledge_documents "
            "(content, category, is_active) VALUES (?, ?, 1)",
            ("legacy knowledge doc", "course_intro"),
        )
        conn.commit()
        conn.close()

        from db.base.schema_migrator import run_migration
        run_migration(str(db_path))

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT source_file, chunk_index, source_hash, chunk_count "
            "FROM knowledge_documents WHERE id=1"
        ).fetchone()
        conn.close()

        assert row[0] is None, f"source_file 应为 NULL,但 = {row[0]!r}"
        assert row[1] is None, f"chunk_index 应为 NULL,但 = {row[1]!r}"
        assert row[2] is None, f"source_hash 应为 NULL,但 = {row[2]!r}"
        assert row[3] is None, f"chunk_count 应为 NULL,但 = {row[3]!r}"


# ----------------------------------------------------------------------
# C. Repository: 新字段持久化 + find_chunks_by_source_hash 排序
# ----------------------------------------------------------------------

class TestRepositoryUploadPersistence:
    def test_add_document_persists_source_file(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="测试内容",
            category="faq",
            embedding=_fake_embed("测试内容"),
            source_file="test.pdf",
        )
        doc = repo.get_document(doc_id)
        assert doc["source_file"] == "test.pdf"

    def test_add_document_persists_chunk_index_and_count(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="chunk 3 内容",
            category="faq",
            embedding=_fake_embed("chunk 3"),
            chunk_index=3,
            chunk_count=10,
        )
        doc = repo.get_document(doc_id)
        assert doc["chunk_index"] == 3
        assert doc["chunk_count"] == 10

    def test_add_document_persists_source_hash(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="hash test",
            category="faq",
            embedding=_fake_embed("hash test"),
            source_hash="deadbeef" * 8,
        )
        doc = repo.get_document(doc_id)
        assert doc["source_hash"] == "deadbeef" * 8

    def test_add_document_optional_upload_fields_default_none(self, in_memory_db):
        """老调用方不传 upload 字段时,允许(可空)。"""
        repo = in_memory_db.knowledge_repo
        doc_id = repo.add_document(
            content="legacy row",
            category="faq",
            embedding=_fake_embed("legacy"),
        )
        doc = repo.get_document(doc_id)
        assert doc["source_file"] is None
        assert doc["chunk_index"] is None
        assert doc["source_hash"] is None
        assert doc["chunk_count"] is None

    def test_find_chunks_by_source_hash_orders_by_index(self, in_memory_db):
        """find_chunks_by_source_hash 按 chunk_index ASC 排序。"""
        repo = in_memory_db.knowledge_repo
        # 故意乱序写入,验证取出顺序
        for i in (2, 0, 1):
            repo.add_document(
                content=f"chunk {i}",
                category="faq",
                embedding=_fake_embed(f"chunk {i}"),
                source_file="doc.pdf",
                chunk_index=i,
                source_hash="abc123",
                chunk_count=3,
            )
        rows = repo.find_chunks_by_source_hash("abc123")
        assert [r["chunk_index"] for r in rows] == [0, 1, 2]
        assert [r["content"] for r in rows] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_find_chunks_by_source_hash_returns_empty_for_unknown(self, in_memory_db):
        repo = in_memory_db.knowledge_repo
        assert repo.find_chunks_by_source_hash("nonexistent") == []

    def test_find_chunks_excludes_soft_deleted(self, in_memory_db):
        """软删除的行不应被 find_chunks 查出。"""
        repo = in_memory_db.knowledge_repo
        for i in range(2):
            repo.add_document(
                content=f"chunk {i}",
                category="faq",
                embedding=_fake_embed(f"chunk {i}"),
                source_file="doc.pdf",
                chunk_index=i,
                source_hash="abc",
                chunk_count=2,
            )
        # 软删除第二个
        repo.delete_document(2, soft_delete=True)
        rows = repo.find_chunks_by_source_hash("abc")
        assert len(rows) == 1
        assert rows[0]["chunk_index"] == 0


# ----------------------------------------------------------------------
# D. DocumentParser: detect_kind + parse_markdown
# ----------------------------------------------------------------------

class TestDocumentParser:
    """document_parser 模块的纯逻辑测试。
    parse_pdf 用 markitdown,真 PDF 解析在 TestDocumentParserPdf 中测。
    """

    def test_detect_kind_pdf_by_extension(self):
        from services.document_parser import detect_kind
        assert detect_kind("notes.pdf", None) == "pdf"
        assert detect_kind("notes.PDF", None) == "pdf"

    def test_detect_kind_markdown_by_extension(self):
        from services.document_parser import detect_kind
        assert detect_kind("README.md", None) == "markdown"
        assert detect_kind("notes.markdown", None) == "markdown"

    def test_detect_kind_markdown_by_content_type(self):
        """扩展名不是 .md 时,看 content-type 兜底。"""
        from services.document_parser import detect_kind
        assert detect_kind("notes.txt", "text/markdown") == "markdown"
        assert detect_kind("notes", "text/markdown") == "markdown"

    def test_detect_kind_pdf_by_content_type(self):
        from services.document_parser import detect_kind
        assert detect_kind("notes.bin", "application/pdf") == "pdf"

    def test_detect_kind_unsupported_raises(self):
        from services.document_parser import detect_kind, UnsupportedFileType
        with pytest.raises(UnsupportedFileType):
            detect_kind("virus.exe", None)
        with pytest.raises(UnsupportedFileType):
            detect_kind("image.png", "image/png")
        with pytest.raises(UnsupportedFileType):
            detect_kind("no_extension", None)

    def test_parse_markdown_decodes_utf8_bom(self):
        """utf-8 BOM 不能污染正文。"""
        from services.document_parser import parse_markdown
        content = "﻿# 标题\n这是正文\n"
        parsed = parse_markdown(content.encode("utf-8"))
        assert parsed.text == "# 标题\n这是正文\n"
        assert parsed.title == "标题"

    def test_parse_markdown_extracts_first_heading_as_title(self):
        from services.document_parser import parse_markdown
        text = "# 数学公式\n## 二次方程\n内容..."
        parsed = parse_markdown(text.encode("utf-8"))
        assert parsed.title == "数学公式"

    def test_parse_markdown_no_heading_title_is_none(self):
        from services.document_parser import parse_markdown
        parsed = parse_markdown("无标题的纯文本。".encode("utf-8"))
        assert parsed.title is None

    def test_parse_empty_bytes_raises_EmptyDocument(self):
        from services.document_parser import (
            parse_markdown, parse, EmptyDocument, UnsupportedFileType,
        )
        with pytest.raises(EmptyDocument):
            parse_markdown(b"")
        with pytest.raises(EmptyDocument):
            parse_markdown(b"   \n\n   ")
        with pytest.raises(EmptyDocument):
            parse(b"", "empty.pdf")

    def test_parse_dispatches_by_kind(self):
        """parse() 顶层入口根据 detect_kind 分发到对应解析器。"""
        from services.document_parser import parse
        kind, parsed = parse("# 标题\n正文".encode("utf-8"), "readme.md")
        assert kind == "markdown"
        assert "正文" in parsed.text

    def test_parse_unsupported_extension_raises(self):
        from services.document_parser import parse, UnsupportedFileType
        with pytest.raises(UnsupportedFileType):
            parse(b"junk", "evil.exe")


# ----------------------------------------------------------------------
# E. TextChunker
# ----------------------------------------------------------------------

class TestTextChunker:
    def test_short_text_yields_single_chunk(self):
        from services.text_chunker import TextChunker
        chunks = TextChunker().split("a" * 500)
        assert len(chunks) == 1
        assert chunks[0].text == "a" * 500
        assert chunks[0].chunk_index == 0

    def test_long_text_yields_multiple_chunks_under_size_limit(self):
        from services.text_chunker import TextChunker
        text = "中文字符。" * 500  # 4 字符 × 500 = 2000 字符
        chunks = TextChunker().split(text)
        assert len(chunks) >= 2
        # 每块不超过 chunk_size + 一些 buffer (langchain 的 chunk_size 是软目标)
        # 使用 chunk_size=200 触发,允许 +/- buffer
        from services.text_chunker import TextChunker, ChunkerSettings
        small_chunks = TextChunker(ChunkerSettings(chunk_size=200, chunk_overlap=50)).split(text)
        assert len(small_chunks) >= 2
        for c in small_chunks:
            assert len(c.text) <= 400  # 200 + 2*100 buffer

    def test_chunk_indices_are_zero_based_and_sequential(self):
        from services.text_chunker import TextChunker, ChunkerSettings
        chunks = TextChunker(ChunkerSettings(chunk_size=200, chunk_overlap=50)).split("a" * 2000)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_text_raises(self):
        from services.text_chunker import TextChunker, EmptyTextError
        with pytest.raises(EmptyTextError):
            TextChunker().split("")

    def test_chunker_settings_defaults(self):
        from services.text_chunker import ChunkerSettings
        s = ChunkerSettings()
        assert s.chunk_size == 500
        assert s.chunk_overlap == 50


# ----------------------------------------------------------------------
# F. Service: add_uploaded_document
# ----------------------------------------------------------------------

class TestServiceUpload:
    def _req(self, file_bytes: bytes, **overrides):
        from services.knowledge_service import UploadRequest
        defaults = dict(
            file_bytes=file_bytes,
            filename="readme.md",
            content_type="text/markdown",
            category="faq",
            keywords=["test"],
            subject="数学",
            grade_level="高中",
            doc_type="course_intro",
            allow_duplicate=False,
        )
        defaults.update(overrides)
        return UploadRequest(**defaults)

    def test_upload_markdown_creates_rows_with_provenance(self, in_memory_db):
        """Markdown 上传 → 多行入库,所有行共享 source_hash,chunk_index 连续。"""
        repo = in_memory_db.knowledge_repo
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        md = "# 高中数学 12 周提分方案\n\n" + "函数、几何、数列。" * 200
        result = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))

        assert result.skipped is False
        assert result.kind == "markdown"
        assert result.chunk_count >= 1
        assert len(result.doc_ids) == result.chunk_count

        # DB 校验
        rows = repo.find_chunks_by_source_hash(result.source_hash)
        assert len(rows) == result.chunk_count
        indices = [r["chunk_index"] for r in rows]
        assert indices == list(range(result.chunk_count))
        for r in rows:
            assert r["source_file"] == "readme.md"
            assert r["source_hash"] == result.source_hash
            assert r["chunk_count"] == result.chunk_count
            assert r["subject"] == "数学"
            assert r["grade_level"] == "高中"
            assert r["category"] == "faq"

    def test_upload_dedup_skips_when_source_hash_exists(self, in_memory_db):
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        md = "# 同一文件\n内容" + "段落。" * 200
        first = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))
        second = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))

        assert first.skipped is False
        assert second.skipped is True
        assert second.doc_ids == first.doc_ids
        assert second.source_hash == first.source_hash

    def test_upload_dedup_can_be_overridden(self, in_memory_db):
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        md = "# dedup 测试\n" + "内容。" * 200
        first = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))
        second = _run(svc.add_uploaded_document(
            self._req(md.encode("utf-8"), allow_duplicate=True)
        ))

        assert first.skipped is False
        assert second.skipped is False
        # 第二次应该新建行,ID 不同
        assert set(second.doc_ids).isdisjoint(set(first.doc_ids))

    def test_upload_increments_faiss_index_size(self, in_memory_db):
        svc = _build_service(in_memory_db)
        _run(svc.initialize())
        before = len(svc.document_ids)

        md = "# faiss 测试\n" + "内容。" * 200
        result = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))

        assert len(svc.document_ids) == before + result.chunk_count

    def test_upload_unsupported_kind_raises(self, in_memory_db):
        svc = _build_service(in_memory_db)
        _run(svc.initialize())
        from services.document_parser import UnsupportedFileType
        with pytest.raises(UnsupportedFileType):
            _run(svc.add_uploaded_document(
                self._req(b"junk", filename="evil.exe", content_type=None)
            ))

    def test_upload_empty_document_raises(self, in_memory_db):
        svc = _build_service(in_memory_db)
        _run(svc.initialize())
        from services.document_parser import EmptyDocument
        with pytest.raises(EmptyDocument):
            _run(svc.add_uploaded_document(self._req(b"   \n\n   ")))

    def test_upload_chunks_are_searchable_after_upload(self, in_memory_db):
        """上传后,搜索应能命中新 chunk。"""
        svc = _build_service(in_memory_db)
        _run(svc.initialize())

        md = "# 一元二次方程 解题套路\n\n" + (
            "判别式、根与系数关系(韦达定理)是中考高频考点。" * 200
        )
        result = _run(svc.add_uploaded_document(self._req(md.encode("utf-8"))))
        # 重建索引以确保新向量可见
        _run(svc.rebuild_index())

        hits = _run(svc.search(
            query="韦达定理", top_k=3, subject="数学", grade_level="高中",
        ))
        assert len(hits) >= 1
        # 至少一个 hit 应该来自本次上传
        hit_ids = {h["id"] for h in hits}
        assert hit_ids & set(result.doc_ids)


# ----------------------------------------------------------------------
# G. API: POST /api/knowledge/upload
# ----------------------------------------------------------------------

class TestApiUpload:
    """端到端 HTTP 测试:multipart/form-data → 200 / 400 / 413 / 422。

    KnowledgeService 默认 db_path 由 tests/conftest.py 强制为 :memory:,
    无需在本类内 monkeypatch。

    控制台分面 B 起 `/api/knowledge` 整个 router 挂 academic / admin 闸门,
    所以本类先显式临时库登录 academic,再打端点。
    """

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from app import app
        from api.core import console_auth as console_dep
        from config.console_auth import CONSOLE_COOKIE_NAME
        from services.console_auth_service import ConsoleAuthService

        console_db = f"sqlite:///{(tmp_path / 'console_knowledge.db').as_posix()}"
        original_init = ConsoleAuthService.__init__

        def _patched_init(self, db_path=None, *args, **kwargs):
            original_init(self, db_path=db_path or console_db)

        monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_init)
        monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)
        ConsoleAuthService(db_path=console_db).ensure_default_users()

        client = TestClient(app)
        login = client.post("/api/console/login",
                            json={"username": "academic", "password": "academic12345"})
        assert login.status_code == 200, login.text
        assert CONSOLE_COOKIE_NAME in login.cookies
        return client

    def _post(self, client, *, filename, content, content_type, **fields):
        return client.post(
            "/api/knowledge/upload",
            files={"file": (filename, content, content_type)},
            data={k: str(v) for k, v in fields.items()},
        )

    def test_upload_markdown_returns_200(self, client):
        md = ("# 数学公式\n\n" + "x^2 + y^2 = z^2." * 200).encode("utf-8")
        r = self._post(
            client, filename="notes.md", content=md,
            content_type="text/markdown",
            category="faq", subject="数学", grade_level="高中",
            doc_type="subject_method",
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["kind"] == "markdown"
        assert data["skipped"] is False
        assert data["chunk_count"] >= 1
        assert len(data["doc_ids"]) == data["chunk_count"]

    def test_upload_unsupported_extension_returns_400(self, client):
        r = self._post(
            client, filename="malware.exe", content=b"x" * 10,
            content_type="application/octet-stream",
            category="faq",
        )
        assert r.status_code == 400

    def test_upload_too_large_returns_413(self, client):
        # 11 MB markdown
        big = (b"# big\n" + b"x" * (11 * 1024 * 1024))
        r = self._post(
            client, filename="big.md", content=big,
            content_type="text/markdown",
            category="faq",
        )
        assert r.status_code == 413

    def test_upload_missing_file_field_returns_422(self, client):
        # 不传 file 字段
        r = client.post(
            "/api/knowledge/upload",
            data={"category": "faq"},
        )
        # FastAPI 对缺少必填 Form/File 返回 422
        assert r.status_code == 422

    def test_upload_missing_category_returns_422(self, client):
        md = b"# test\ncontent"
        r = self._post(
            client, filename="t.md", content=md,
            content_type="text/markdown",
        )
        # category 缺 → 422
        assert r.status_code == 422

    # 注:去重行为(同一 source_hash 二次上传返回 skipped=True)
    # 已在 TestServiceUpload.test_upload_dedup_skips_when_source_hash_exists
    # 覆盖。API 层无法用同一 client 跨请求验证,因为 endpoint 每次创建
    # 全新 KnowledgeService + in-memory DB,跨请求状态会被擦掉。