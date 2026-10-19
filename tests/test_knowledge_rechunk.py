"""存量 chunk 重切脚本(scripts/rechunk_knowledge.py)单元测试。

覆盖:
9.  还原算法: 人造 chunk(有 overlap) → 还原全文与原始文本一致(overlap 正确消解)
10. 重入库幂等: 同一份内容执行两次, chunk_count 与行数不变
+  安全约束: 生产库路径被拒绝 / dry-run 不写库 / 精确切片可逐字还原

全部离线: 不调 embedding API(补 embedding 为 NULL), 不碰生产库。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

rechunk = importlib.import_module("scripts.rechunk_knowledge")


# ----------------------------------------------------------------------
# 9. 还原算法
# ----------------------------------------------------------------------

class TestFindOverlap:
    def test_finds_longest_suffix_prefix(self):
        assert rechunk.find_overlap("abcXYZ", "XYZdef") == 3

    def test_no_match_returns_zero(self):
        assert rechunk.find_overlap("abc", "def") == 0

    def test_respects_limit(self):
        """真实重合长度超过 limit 时按 limit 截断(不越界去凑更长的匹配)。"""
        # 真实重合 4 字("bbbb"); limit=4 能拿全, limit=2 只能拿 2
        assert rechunk.find_overlap("aaaabbbb", "bbbbcccc", limit=4) == 4
        assert rechunk.find_overlap("aaaabbbb", "bbbbcccc", limit=2) == 2
        # 重合只有 1 字, 放多宽的 limit 都只拿 1
        assert rechunk.find_overlap("xyzQ", "Qrst", limit=200) == 1

    def test_empty_inputs(self):
        assert rechunk.find_overlap("", "abc") == 0
        assert rechunk.find_overlap("abc", "") == 0


class TestStitchChunks:
    def test_restores_original_text_from_overlapping_chunks(self):
        """3 个人造 chunk(有重叠) → 还原出的全文与原始文本一致。"""
        original = "".join(f"第{i}段内容。" for i in range(1, 41))  # ~240 字
        # 手工按 100 字切、重叠 30 字, 模拟旧参数切出的存量 chunk
        size, overlap = 100, 30
        chunks: List[str] = []
        start = 0
        while start < len(original):
            chunks.append(original[start:start + size])
            if start + size >= len(original):
                break
            start += size - overlap
        assert len(chunks) >= 3, "构造样本应至少 3 块"

        assert rechunk.stitch_chunks(chunks, overlap) == original

    def test_single_chunk_is_identity(self):
        assert rechunk.stitch_chunks(["唯一内容"], 200) == "唯一内容"

    def test_empty_list_returns_empty_string(self):
        assert rechunk.stitch_chunks([], 200) == ""

    def test_adjacent_only_comparison(self):
        """只用相邻块比对: 中间块与首块共享的巧合重复不应被误当成 overlap。

        首块 "ABABA"、第二块 "BABAC"、第三块 "BACAD"。
        正确结果: ABABA + C + AD = "ABABACAD"。
        若拿累积文本去比, 会把第二块尾部(与首块重复的 "ABA")也多切掉。
        """
        assert rechunk.stitch_chunks(["ABABA", "BABAC", "BACAD"], 4) == "ABABACAD"


# ----------------------------------------------------------------------
# 精确切片(重切的正确性根基)
# ----------------------------------------------------------------------

class TestRechunkExact:
    def test_slices_concatenate_back_to_text(self):
        """重切结果按 overlap 拼接必须逐字还原原文(否则重复执行会漂移)。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        text = ("这是一个用于验证重切可逆性的长文本。" * 40) + "\n\n结尾段落"
        chunker = TextChunker(ChunkerSettings(chunk_size=100, chunk_overlap=20))
        slices = rechunk.rechunk_exact(chunker, text, 20)
        assert len(slices) > 1
        assert rechunk.stitch_chunks(slices, 20) == text

    def test_preserves_separators_dropped_by_splitter(self):
        """splitter 会吞掉块边界的空白 run; 精确切片必须把它保住。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        text = "\n\n".join(f"段落{i}" for i in range(60))
        chunker = TextChunker(ChunkerSettings(chunk_size=50, chunk_overlap=10))
        slices = rechunk.rechunk_exact(chunker, text, 10)
        assert rechunk.stitch_chunks(slices, 10) == text
        # 直接拼接 splitter 输出则会丢字符 —— 这正是不能用它的原因
        naive = "".join(c.text for c in chunker.split(text))
        assert naive != text

    def test_empty_text_raises(self):
        from services.text_chunker import ChunkerSettings, TextChunker

        chunker = TextChunker(ChunkerSettings(chunk_size=100, chunk_overlap=20))
        with pytest.raises(Exception):
            rechunk.rechunk_exact(chunker, "", 20)


# ----------------------------------------------------------------------
# 10. 重入库幂等
# ----------------------------------------------------------------------

def _seed_temp_db(db_url: str, chunks: List[str], source_hash: str = "deadbeef",
                  source_file: str = "demo.md", embedding=None) -> None:
    """向临时库写入一组 chunk(embedding 默认留空, 免联网)。"""
    from db.base.session_manager import SessionManager
    from db.models import KnowledgeDocument

    sm = SessionManager(db_path=db_url)
    with sm.session_scope() as session:
        for index, text in enumerate(chunks):
            session.add(KnowledgeDocument(
                content=text,
                category="subject_method",
                keywords=["数学"],
                embedding=embedding,
                subject="数学",
                grade_level="初中",
                doc_type="subject_method",
                source_file=source_file,
                chunk_index=index,
                source_hash=source_hash,
                chunk_count=len(chunks),
            ))
    sm.close()


def _read_rows(db_url: str):
    from db.base.session_manager import SessionManager
    from db.models import KnowledgeDocument

    sm = SessionManager(db_path=db_url)
    with sm.session_scope() as session:
        rows = (
            session.query(KnowledgeDocument)
            .filter(KnowledgeDocument.source_hash.isnot(None))
            .order_by(KnowledgeDocument.chunk_index.asc())
            .all()
        )
        out = [
            {
                "id": r.id, "content": r.content, "chunk_index": r.chunk_index,
                "chunk_count": r.chunk_count, "source_hash": r.source_hash,
                "source_file": r.source_file, "embedding": r.embedding,
            }
            for r in rows
        ]
    sm.close()
    return out


@pytest.fixture
def temp_db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'rechunk.db'}"


class TestRechunkIdempotency:
    def _run_apply(self, db_url: str) -> int:
        argv = ["--db", db_url, "--apply"]
        return rechunk.run(rechunk.parse_args(argv))

    def _run_dry(self, db_url: str) -> int:
        return rechunk.run(rechunk.parse_args(["--db", db_url]))

    def test_apply_twice_is_idempotent(self, temp_db_url, capsys):
        """同一份内容执行两次, chunk_count 与行数不变。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        original = "".join(f"第{i}段内容，用于验证幂等。" for i in range(1, 61))
        old_chunker = TextChunker(ChunkerSettings(chunk_size=200, chunk_overlap=40))
        old_chunks = [c.text for c in old_chunker.split(original)]
        assert len(old_chunks) > 1

        _seed_temp_db(temp_db_url, old_chunks)

        assert self._run_apply(temp_db_url) == 0
        after_first = _read_rows(temp_db_url)
        assert after_first
        first_count = after_first[0]["chunk_count"]
        assert first_count == len(after_first)

        assert self._run_apply(temp_db_url) == 0
        after_second = _read_rows(temp_db_url)

        assert len(after_second) == len(after_first), "第二次执行行数不应变化"
        assert after_second[0]["chunk_count"] == first_count, "chunk_count 不应变化"
        assert [r["content"] for r in after_second] == [
            r["content"] for r in after_first
        ], "第二次执行的 chunk 内容应与第一次完全一致"

    def test_apply_then_dry_run_chunk_counts_match(self, temp_db_url, capsys):
        """第二次执行报告的「旧 chunk 数 → 新 chunk 数」两边相等(无漂移)。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        original = "".join(f"段落{i}。" for i in range(1, 201))
        old_chunks = [
            c.text for c in TextChunker(
                ChunkerSettings(chunk_size=200, chunk_overlap=40)
            ).split(original)
        ]
        _seed_temp_db(temp_db_url, old_chunks)

        assert self._run_apply(temp_db_url) == 0
        capsys.readouterr()
        assert self._run_dry(temp_db_url) == 0
        out = capsys.readouterr().out
        assert "demo.md: 旧 chunk 数" in out
        # 形如 "旧 chunk 数 N → 新 chunk 数 N"
        line = next(ln for ln in out.splitlines() if "demo.md: 旧 chunk" in ln)
        nums = [int(tok) for tok in line.replace("→", " ").split() if tok.isdigit()]
        assert len(nums) == 2 and nums[0] == nums[1], f"应无漂移: {line}"

    def test_dry_run_does_not_write(self, temp_db_url, capsys):
        """默认 dry-run: 行数与内容都不变。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        original = "".join(f"内容{i}。" for i in range(1, 201))
        old_chunks = [
            c.text for c in TextChunker(
                ChunkerSettings(chunk_size=200, chunk_overlap=40)
            ).split(original)
        ]
        _seed_temp_db(temp_db_url, old_chunks)
        before = _read_rows(temp_db_url)

        assert self._run_dry(temp_db_url) == 0
        after = _read_rows(temp_db_url)
        assert [r["id"] for r in after] == [r["id"] for r in before]
        assert [r["content"] for r in after] == [r["content"] for r in before]

    def test_reports_old_to_new_chunk_counts(self, temp_db_url, capsys):
        """输出报告必须含 每个文件 旧 chunk 数 → 新 chunk 数。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        original = "".join(f"内容{i}。" for i in range(1, 301))
        old_chunks = [
            c.text for c in TextChunker(
                ChunkerSettings(chunk_size=200, chunk_overlap=40)
            ).split(original)
        ]
        _seed_temp_db(temp_db_url, old_chunks, source_file="物理.md")

        assert self._run_dry(temp_db_url) == 0
        out = capsys.readouterr().out
        assert "物理.md" in out
        assert "旧 chunk 数" in out and "新 chunk 数" in out


# ----------------------------------------------------------------------
# 安全约束
# ----------------------------------------------------------------------

class TestProductionGuard:
    def test_prod_db_without_flag_is_rejected(self, capsys):
        prod_url = f"sqlite:///{PROJECT_ROOT / 'data' / 'smart_appointment.db'}"
        assert rechunk.run(rechunk.parse_args(["--db", prod_url])) == 2
        assert "拒绝执行" in capsys.readouterr().err

    def test_prod_db_flag_is_recognized(self):
        prod_url = f"sqlite:///{PROJECT_ROOT / 'data' / 'smart_appointment.db'}"
        args = rechunk.parse_args(["--db", prod_url, "--i-know-this-is-prod"])
        assert args.confirm_prod is True

    def test_is_prod_db_detection(self, tmp_path):
        prod_url = f"sqlite:///{PROJECT_ROOT / 'data' / 'smart_appointment.db'}"
        assert rechunk._is_prod_db(prod_url) is True
        assert rechunk._is_prod_db(f"sqlite:///{tmp_path / 'other.db'}") is False
        assert rechunk._is_prod_db("sqlite:///:memory:") is False

    def test_source_dir_nonexistent_falls_back_to_stitch(self, temp_db_url):
        """--source-dir 里找不到原文件时退回拼接还原, 不报错。"""
        from services.text_chunker import ChunkerSettings, TextChunker

        original = "".join(f"内容{i}。" for i in range(1, 201))
        old_chunks = [
            c.text for c in TextChunker(
                ChunkerSettings(chunk_size=200, chunk_overlap=40)
            ).split(original)
        ]
        _seed_temp_db(temp_db_url, old_chunks)
        code = rechunk.run(rechunk.parse_args([
            "--db", temp_db_url, "--source-dir", str(PROJECT_ROOT / "does_not_exist"),
        ]))
        assert code == 0


class TestUnrepairableReporting:
    def test_missing_chunk_index_is_reported_not_silently_skipped(
        self, temp_db_url, capsys
    ):
        """chunk_index 缺失 → 记入「无法还原」清单, 不静默跳过。"""
        _seed_temp_db(temp_db_url, ["只有一块但序号未知"])
        from db.base.session_manager import SessionManager
        from db.models import KnowledgeDocument

        sm = SessionManager(db_path=temp_db_url)
        with sm.session_scope() as session:
            row = session.query(KnowledgeDocument).first()
            row.chunk_index = None
        sm.close()

        assert rechunk.run(rechunk.parse_args(["--db", temp_db_url])) == 0
        out = capsys.readouterr().out
        assert "无法还原的文件" in out
        assert "chunk_index 缺失" in out


# ----------------------------------------------------------------------
# P0-1 / P0-2 / P0-3: embedding 预检守卫 + 中途失败回滚 + 文案修正
# ----------------------------------------------------------------------

OLD_EMBEDDING = [0.5] * 4  # 存量行的「有向量」标记, 用来区分旧行与新写的 NULL 行


def _old_chunks(paragraphs: int = 201, size: int = 200, overlap: int = 40) -> List[str]:
    """造一批旧参数(200/40)切出的 chunk, 供 seed 用。"""
    from services.text_chunker import ChunkerSettings, TextChunker

    original = "".join(f"内容{i}。" for i in range(1, paragraphs + 1))
    return [
        c.text for c in TextChunker(
            ChunkerSettings(chunk_size=size, chunk_overlap=overlap)
        ).split(original)
    ]


def _install_embedding(monkeypatch, fail_after=None, error=None) -> List[str]:
    """把 services.text_embedding.embed_input 换成可控替身, 返回调用记录。

    fail_after=None 表示一直成功; 否则第 fail_after+1 次调用起抛 error
    (第 1 次通常是预检探针)。脚本内部是 `from services.text_embedding import
    embed_input` 的延迟导入, 所以 patch 模块属性即可生效。
    """
    from services import text_embedding

    calls: List[str] = []
    error = error or RuntimeError("Missing credentials")

    def _fake(text: str):
        calls.append(text)
        if fail_after is not None and len(calls) > fail_after:
            raise error
        return [0.1] * 4

    monkeypatch.setattr(text_embedding, "embed_input", _fake)
    return calls


@pytest.fixture(autouse=True)
def offline_embedding(monkeypatch):
    """全模块默认离线: embedding 换成固定向量。

    既让既有用例不联网, 也让 --apply 的 embedding 预检有确定结果
    (需要制造失败的用例在自己的函数体里再覆盖一次)。
    """
    _install_embedding(monkeypatch)


class TestEmbeddingPrecheck:
    """P0-1: --apply 前先探针 embedding, 失败默认拒绝写库(exit 3, 0 写)。"""

    def test_apply_precheck_failure_refuses_with_zero_writes(
        self, temp_db_url, capsys, monkeypatch
    ):
        _install_embedding(monkeypatch, fail_after=0)  # 连预检都失败
        _seed_temp_db(temp_db_url, _old_chunks(), embedding=OLD_EMBEDDING)
        before = _read_rows(temp_db_url)
        assert before

        code = rechunk.run(rechunk.parse_args(["--db", temp_db_url, "--apply"]))

        assert code == 3, "embedding 不可用时 --apply 必须 exit 3"
        after = _read_rows(temp_db_url)
        assert [r["id"] for r in after] == [r["id"] for r in before], "不得删旧行"
        assert [r["content"] for r in after] == [r["content"] for r in before]
        assert [r["embedding"] for r in after] == [
            r["embedding"] for r in before
        ], "不得写任何 NULL 行"

    def test_precheck_failure_message_explains_both_paths_and_way_out(
        self, temp_db_url, capsys, monkeypatch
    ):
        _install_embedding(monkeypatch, fail_after=0)
        _seed_temp_db(temp_db_url, _old_chunks(), embedding=OLD_EMBEDDING)

        assert rechunk.run(
            rechunk.parse_args(["--db", temp_db_url, "--apply"])
        ) == 3
        err = capsys.readouterr().err
        assert "两条检索路径下都召回不到" in err
        assert "--allow-null-embedding" in err, "必须给出出路"
        assert "EMBEDDING_PROVIDER" in err and "EMBEDDING_API_KEY" in err
        assert "BM25 仍可召回" not in err

    def test_allow_null_embedding_skips_precheck_and_writes_null(
        self, temp_db_url, capsys, monkeypatch
    ):
        """带开关: 放行 NULL 写入, 输出含新告警且不含错误的「BM25 仍可召回」。"""
        calls = _install_embedding(monkeypatch, fail_after=0)
        _seed_temp_db(temp_db_url, _old_chunks(), embedding=OLD_EMBEDDING)
        before = _read_rows(temp_db_url)

        code = rechunk.run(rechunk.parse_args(
            ["--db", temp_db_url, "--apply", "--allow-null-embedding"]
        ))

        assert code == 0
        assert rechunk.EMBEDDING_PROBE_TEXT not in calls, "带开关时不应再做预检探针"
        after = _read_rows(temp_db_url)
        assert [r["id"] for r in after] != [r["id"] for r in before], "应重写该组"
        assert after, "应写入新行"
        assert all(r["embedding"] is None for r in after), "宽松模式写 NULL"

        out = capsys.readouterr().out
        assert "BM25 仍可召回" not in out
        assert "两条检索路径下都召回不到" in out
        assert "写 NULL" in out and str(len(after)) in out

    def test_allow_null_embedding_is_noop_without_apply(
        self, temp_db_url, capsys, monkeypatch
    ):
        """开关只对 --apply 有意义; dry-run 仍不写库也不探针。"""
        calls = _install_embedding(monkeypatch, fail_after=0)
        _seed_temp_db(temp_db_url, _old_chunks(), embedding=OLD_EMBEDDING)
        before = _read_rows(temp_db_url)

        code = rechunk.run(rechunk.parse_args(
            ["--db", temp_db_url, "--allow-null-embedding"]
        ))

        assert code == 0
        assert calls == [], "dry-run 不应调用 embedding"
        assert [r["id"] for r in _read_rows(temp_db_url)] == [
            r["id"] for r in before
        ]


class TestMidRunFailureRollsBack:
    """P0-2: 预检过后某次 embed 抛错 → 该文件事务整体回滚, 不留半截 NULL。"""

    def test_failed_group_rolls_back_and_is_reported(self, temp_db_url, capsys, monkeypatch):
        _seed_temp_db(temp_db_url, _old_chunks(), source_hash="aaa", source_file="a.md",
                      embedding=OLD_EMBEDDING)
        _seed_temp_db(temp_db_url, _old_chunks(), source_hash="bbb", source_file="b.md",
                      embedding=OLD_EMBEDDING)
        # 第 1 次是预检探针, 第 2 次(a.md 首块)成功, 第 3 次起失败
        _install_embedding(monkeypatch, fail_after=2)
        before = _read_rows(temp_db_url)

        code = rechunk.run(rechunk.parse_args(["--db", temp_db_url, "--apply"]))

        assert code == 0, "单文件失败不应让整批返回非 0"
        after = _read_rows(temp_db_url)
        assert [r["id"] for r in after] == [r["id"] for r in before], "旧行应原样保留"
        assert all(r["embedding"] is not None for r in after), "不得残留 NULL 行"
        assert {r["source_hash"] for r in after} == {"aaa", "bbb"}

        out = capsys.readouterr().out
        assert "无法还原的文件 2 个" in out
        assert "a.md" in out and "b.md" in out
        assert "已回滚" in out

    def test_rollback_is_per_file_not_whole_batch(self, temp_db_url, capsys, monkeypatch):
        """前一个文件的失败不得带走后一个文件: b.md 依然成功写库。"""
        _seed_temp_db(temp_db_url, ["a.md 的旧内容"], source_hash="aaa",
                      source_file="a.md", embedding=OLD_EMBEDDING)
        _seed_temp_db(temp_db_url, _old_chunks(), source_hash="bbb", source_file="b.md",
                      embedding=OLD_EMBEDDING)
        from services import text_embedding

        state = {"failed_for_a": False}

        def _fake(text: str):
            # 只在处理 a.md 的内容时失败一次, b.md 恢复正常
            if "a.md" in text and not state["failed_for_a"]:
                state["failed_for_a"] = True
                raise RuntimeError("boom")
            return [0.1] * 4

        monkeypatch.setattr(text_embedding, "embed_input", _fake)

        assert rechunk.run(rechunk.parse_args(["--db", temp_db_url, "--apply"])) == 0
        after = _read_rows(temp_db_url)
        a_rows = [r for r in after if r["source_file"] == "a.md"]
        b_rows = [r for r in after if r["source_file"] == "b.md"]

        assert len(a_rows) == 1 and a_rows[0]["embedding"] == OLD_EMBEDDING, \
            "a.md 事务回滚, 旧行原样保留"
        assert a_rows[0]["content"] == "a.md 的旧内容", "a.md 未被重写"
        assert len(b_rows) > 1, "b.md 应被重新分块"
        assert all(r["embedding"] == [0.1] * 4 for r in b_rows), "b.md 正常写入"

        out = capsys.readouterr().out
        assert "a.md" in out and "无法还原的文件" in out
        assert "b.md" not in out.split("无法还原的文件")[1], "b.md 不应出现在失败清单"


class TestEmbedderModes:
    """make_embedder 的严格/宽松两种模式(纯函数级)。"""

    def test_strict_mode_raises_instead_of_returning_none(self, monkeypatch):
        _install_embedding(monkeypatch, fail_after=0)
        embed_fn, stats = rechunk.make_embedder()
        with pytest.raises(Exception):
            embed_fn("任意文本")
        assert stats["ok"] == 0 and stats["null"] == 0, "严格模式不计 NULL"

    def test_strict_mode_keeps_raising_after_first_failure(self, monkeypatch):
        _install_embedding(monkeypatch, fail_after=0)
        embed_fn, _ = rechunk.make_embedder()
        for _ in range(2):
            with pytest.raises(Exception):
                embed_fn("任意文本")

    def test_lenient_mode_returns_none_and_reports_once(self, monkeypatch, capsys):
        _install_embedding(monkeypatch, fail_after=0)
        embed_fn, stats = rechunk.make_embedder(allow_null=True)

        assert embed_fn("第一块") is None
        assert embed_fn("第二块") is None
        assert stats["null"] == 2 and stats["ok"] == 0

        out = capsys.readouterr().out
        assert out.count("embedding 不可用") == 1, "失败原因只打印一次, 不逐行刷屏"
        assert "两条检索路径下都召回不到" in out

    def test_lenient_mode_counts_successes(self, monkeypatch):
        _install_embedding(monkeypatch)
        embed_fn, stats = rechunk.make_embedder(allow_null=True)
        assert embed_fn("文本") == [0.1] * 4
        assert stats == {"ok": 1, "null": 0}

    def test_empty_vector_from_provider_counts_as_unavailable(self, monkeypatch):
        from services import text_embedding

        monkeypatch.setattr(text_embedding, "embed_input", lambda text: [])
        assert rechunk.check_embedding_available() is not None

    def test_probe_returns_none_when_provider_ok(self, monkeypatch):
        _install_embedding(monkeypatch)
        assert rechunk.check_embedding_available() is None


class TestNullChunkMessage:
    """P0-3: 「BM25 仍可召回」是错的, 脚本里不得再出现。"""

    def test_script_source_never_claims_bm25_still_recalls(self):
        source = (PROJECT_ROOT / "scripts" / "rechunk_knowledge.py").read_text(
            encoding="utf-8"
        )
        assert "BM25 仍可召回" not in source

    def test_warning_constant_states_both_paths_miss(self):
        assert "hybrid" in rechunk.NULL_CHUNK_RETRIEVAL_WARNING
        assert "dense" in rechunk.NULL_CHUNK_RETRIEVAL_WARNING
        assert "都召回不到" in rechunk.NULL_CHUNK_RETRIEVAL_WARNING
        assert "分路之前" in rechunk.NULL_CHUNK_RETRIEVAL_WARNING
        assert "BM25 仍可召回" not in rechunk.NULL_CHUNK_RETRIEVAL_WARNING

    def test_docstring_uses_accurate_wording(self):
        doc = rechunk.__doc__ or ""
        assert "都召回不到" in doc
        assert "BM25 仍可召回" not in doc


class TestGuardInteraction:
    """P0-1/P0-2 开关与既有生产库守卫互不干扰。"""

    PROD_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'smart_appointment.db'}"

    def test_prod_guard_still_exit_2_without_flag(self, capsys):
        assert rechunk.run(rechunk.parse_args(["--db", self.PROD_URL, "--apply"])) == 2
        assert "拒绝执行" in capsys.readouterr().err

    def test_prod_guard_not_bypassed_by_allow_null_embedding(self, capsys):
        code = rechunk.run(rechunk.parse_args(
            ["--db", self.PROD_URL, "--apply", "--allow-null-embedding"]
        ))
        assert code == 2, "新开关不得绕过生产库守卫"
        assert "拒绝执行" in capsys.readouterr().err

    def test_prod_guard_runs_before_embedding_precheck(self, temp_db_url, capsys, monkeypatch):
        """两个开关互不干扰: 生产库守卫先判(exit 2), 预检根本不会跑到。

        用 monkeypatch 把临时库伪装成生产库, 避免真去开生产库连接。
        """
        calls = _install_embedding(monkeypatch, fail_after=0)
        monkeypatch.setattr(rechunk, "_is_prod_db", lambda url: True)

        code = rechunk.run(rechunk.parse_args([
            "--db", temp_db_url, "--apply", "--allow-null-embedding",
        ]))

        assert code == 2
        assert calls == [], "生产库守卫之前不应调用 embedding"
        assert "拒绝执行" in capsys.readouterr().err

    def test_allow_null_embedding_flag_parsed(self):
        args = rechunk.parse_args(["--db", "sqlite:///x.db", "--allow-null-embedding"])
        assert args.allow_null_embedding is True
        assert rechunk.parse_args(["--db", "sqlite:///x.db"]).allow_null_embedding is False
