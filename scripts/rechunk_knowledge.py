"""存量知识库 chunk 重切脚本(维护用,不是测试)。

背景: services/text_chunker.py 的分块默认值从 1000/200 收紧到 500/50
(与 config/components.yaml 的 splitter 声明对齐; 500 中文字 ≈ 500 token < 512,
精排 cross-encoder 不再截断 chunk 后半段)。存量行仍是旧参数切出来的,
本脚本把它们按新参数重切回库。

还原: 无原文件时按 source_hash 分组, 按 chunk_index ASC 拼接,
相邻两块用最长后缀-前缀匹配(上限 OVERLAP_SEEK_LIMIT 字)消解 overlap。
有 --source-dir <dir> 时优先直接读原文件重切。

为什么重切不是「直接 split() 的返回值」:
    RecursiveCharacterTextSplitter 会吞掉落在块边界上的分隔符(实测: README
    7020 字重切后拼接只剩 6999 字, 丢的正是 "\n\n" / "\n" 这类空白 run)。
    因此 split() 的输出天然不可逆 —— 直接入库的话, 下次再还原会得到又短一点的
    文本, 反复执行会持续漂移, 无法满足「第二次 --apply 结果不变」。
    这里改为: 取 splitter 的**边界决策**(每块结束位置), 按 chunk_overlap
    重新切成原文的**精确切片**。切片之间是严格 overlap 关系, 拼接必然逐字还原原文,
    于是「还原 → 重切」成为幂等投影, 重复执行结果恒定。

安全约束:
  - **默认 dry-run**, 只有显式 --apply 才写库;
  - --db 指向仓库内 data/smart_appointment.db 时必须同时给
    --i-know-this-is-prod, 否则拒绝执行(exit 2);
  - --apply 前先做一次 embedding 预检, 失败就**一行都不写**(exit 3) ——
    默认绝不写 NULL 行: NULL 块在 hybrid / dense 两条检索路径下都召回不到
    (见 NULL_CHUNK_RETRIEVAL_WARNING), 写进去等于把整库内容变成孤儿行。
    确认接受「暂时检索不到」时才显式加 --allow-null-embedding;
  - 写库中途 embedding 失败时, 该 source_hash 的事务整体回滚, 不留半截 NULL,
    该文件记入 unrepaired 报告后继续处理下一个文件;
  - 单个 source_hash 的「删旧行 + 写新行」在**同一事务**里完成,
    任一环节抛错整体回滚, 不留半截状态。

用法:
  ./.venv/Scripts/python.exe scripts/rechunk_knowledge.py --db "sqlite:///.../copy.db"          # dry-run
  ./.venv/Scripts/python.exe scripts/rechunk_knowledge.py --db "sqlite:///.../copy.db" --apply  # 写库
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppSettings  # noqa: E402
from db.base.session_manager import SessionManager  # noqa: E402
from db.models import KnowledgeDocument  # noqa: E402
from services import document_parser  # noqa: E402
from services.text_chunker import ChunkerSettings, TextChunker  # noqa: E402

# 还原时相邻两块最多往前找多少字来匹配 overlap(规格给的上限)
OVERLAP_SEEK_LIMIT = 200

PROD_DB_RELATIVE = Path("data") / "smart_appointment.db"

# embedding 预检用的短文本(不落库, 只探一次 provider 是否可用)
EMBEDDING_PROBE_TEXT = "embedding 预检"

# exit code 约定: 0 正常 / 2 生产库守卫 / 3 embedding 不可用
EXIT_EMBEDDING_UNAVAILABLE = 3

# NULL 块的准确后果(防回归: 早期文案曾声称 NULL 块仍会被 BM25 召回, 是错的)。
# services/hybrid_retriever.py 在**分路之前**就按 embedding 非空过滤候选集
# (candidates = [c for c in candidates if c.get("embedding") ...]), dense 与
# BM25 用的是同一批候选, 而 dense 路径只认 FAISS 索引里的向量 ——
# 因此 NULL 块在 hybrid 与 dense 两条路径下都召回不到。
NULL_CHUNK_RETRIEVAL_WARNING = (
    "NULL 块在 hybrid / dense 两条检索路径下都召回不到"
    "(候选集在分路之前即按 embedding 非空过滤), "
    "需修好 embedding 后重跑, 或在服务启动补算向量后才可召回"
)

EmbedFn = Callable[[str], Optional[List[float]]]


# ----------------------------------------------------------------------
# 还原算法(纯函数, 可单测)
# ----------------------------------------------------------------------

def find_overlap(left: str, right: str, limit: int = OVERLAP_SEEK_LIMIT) -> int:
    """最长后缀-前缀匹配: 返回 left 末尾与 right 开头重合的字符数。

    从长到短试, limit 内一个字都不匹配时返回 0。
    """
    max_len = min(limit, len(left), len(right))
    for n in range(max_len, 0, -1):
        if left[-n:] == right[:n]:
            return n
    return 0


def stitch_chunks(chunks: List[str], limit: int = OVERLAP_SEEK_LIMIT) -> str:
    """把有序 chunk 拼回全文, 消解相邻块之间的 overlap。

    只比**相邻两块**(prev 的后缀 vs next 的前缀)。拿累积文本去比会匹配到
    更早内容的巧合重复, 反而多切掉正文。
    """
    if not chunks:
        return ""
    text = chunks[0]
    prev = chunks[0]
    for nxt in chunks[1:]:
        text += nxt[find_overlap(prev, nxt, limit):]
        prev = nxt
    return text


# ----------------------------------------------------------------------
# 重切(精确切片)
# ----------------------------------------------------------------------

def chunk_boundaries(pieces: List[str], text: str) -> List[int]:
    """把 splitter 的输出映射回它在原文中的结束下标。

    pieces 是 text 的**保序子串序列**(分隔符可能被吞掉、相邻块会重叠),
    因此从上一块的**起点之后**开始找 —— 用上一块的结束位置做游标会因为重叠
    而找不到下一块。找不到说明假设不成立, 抛错让上层记入「无法还原」。

    收尾: splitter 会吞掉文末的空白 run, 末块补到 len(text), 否则尾部字符丢失,
    可逆性自检会误判为失败。
    """
    ends: List[int] = []
    search_from = 0
    for piece in pieces:
        start = text.find(piece, search_from)
        if start < 0:
            raise ValueError("splitter 输出不是原文的保序子串, 无法定位边界")
        search_from = start + 1
        ends.append(start + len(piece))
    if ends and ends[-1] < len(text):
        ends[-1] = len(text)
    return ends


def rechunk_exact(
    chunker: TextChunker, text: str, chunk_overlap: int
) -> List[str]:
    """按 new params 重切, 返回**原文的精确切片**(可逐字拼回)。

    用 splitter 决定「每块切到哪」, 但块内容取自原文切片, 从而不丢分隔符:
        第 0 块 = text[0:ends[0]]
        第 i 块 = text[ends[i-1] - overlap : ends[i]]
    拼接时相邻块恰好重叠 overlap 字, 且 ends[-1] == len(text) 时拼接结果 == text。
    """
    pieces = [c.text for c in chunker.split(text)]
    if not pieces:
        return []
    ends = chunk_boundaries(pieces, text)
    slices: List[str] = []
    prev_start = 0
    for index, end in enumerate(ends):
        start = 0 if index == 0 else max(prev_start + 1, ends[index - 1] - chunk_overlap)
        if end <= start:
            raise ValueError("分块边界非法(块长 <= 重叠), 请调小 CHUNK_OVERLAP")
        slices.append(text[start:end])
        prev_start = start
    return slices


# ----------------------------------------------------------------------
# 存量行读取
# ----------------------------------------------------------------------

def load_groups(sm: SessionManager) -> List[Dict[str, Any]]:
    """按 source_hash 聚合存量 chunk(排除软删除与无 source_hash 的行)。"""
    with sm.session_scope() as session:
        rows = (
            session.query(KnowledgeDocument)
            .filter(KnowledgeDocument.is_active == 1)
            .filter(KnowledgeDocument.source_hash.isnot(None))
            .all()
        )
        grouped: Dict[str, List[KnowledgeDocument]] = defaultdict(list)
        for row in rows:
            grouped[row.source_hash].append(row)

        groups: List[Dict[str, Any]] = []
        for source_hash, docs in grouped.items():
            docs = sorted(
                docs,
                key=lambda d: (d.chunk_index if d.chunk_index is not None else 0),
            )
            groups.append({
                "source_hash": source_hash,
                "source_file": docs[0].source_file,
                "category": docs[0].category,
                "keywords": docs[0].keywords or [],
                "subject": docs[0].subject,
                "grade_level": docs[0].grade_level,
                "doc_type": docs[0].doc_type,
                "ids": [d.id for d in docs],
                "contents": [d.content for d in docs],
                "old_chunk_count": len(docs),
                "chunk_index_missing": any(d.chunk_index is None for d in docs),
            })
    return sorted(groups, key=lambda g: g["source_file"] or g["source_hash"])


def _read_source_file(source_dir: Path, source_file: Optional[str]) -> Optional[str]:
    """--source-dir 下按文件名找原文件并解析; 找不到返回 None(上层退回拼接)。"""
    if not source_file:
        return None
    candidate = Path(source_dir) / source_file
    if not candidate.is_file():
        return None
    try:
        _kind, parsed = document_parser.parse(
            candidate.read_bytes(), candidate.name,
        )
    except Exception as exc:  # noqa: BLE001 - 单个文件解析失败不应中断整批
        print(f"  [!] 原文件解析失败, 退回拼接还原: {source_file}: {exc}")
        return None
    return parsed.text


# ----------------------------------------------------------------------
# 写入(事务内 删旧 + 写新)
# ----------------------------------------------------------------------

def apply_group(
    sm: SessionManager,
    group: Dict[str, Any],
    new_texts: List[str],
    embed_fn: EmbedFn,
) -> List[int]:
    """单事务: 按 (source_hash + source_file) 删旧行, 再写入新 chunk。"""
    source_hash = group["source_hash"]
    source_file = group["source_file"]
    new_ids: List[int] = []
    with sm.session_scope() as session:
        query = session.query(KnowledgeDocument).filter(
            KnowledgeDocument.source_hash == source_hash
        )
        if source_file is None:
            query = query.filter(KnowledgeDocument.source_file.is_(None))
        else:
            query = query.filter(KnowledgeDocument.source_file == source_file)
        for row in query.all():
            session.delete(row)
        session.flush()

        total = len(new_texts)
        for index, text in enumerate(new_texts):
            doc = KnowledgeDocument(
                content=text,
                category=group["category"],
                keywords=group["keywords"],
                embedding=embed_fn(text),
                subject=group["subject"],
                grade_level=group["grade_level"],
                doc_type=group["doc_type"],
                source_file=source_file,
                chunk_index=index,
                source_hash=source_hash,
                chunk_count=total,
            )
            session.add(doc)
            session.flush()
            new_ids.append(doc.id)
    return new_ids


def make_embedder(allow_null: bool = False) -> Tuple[EmbedFn, Dict[str, int]]:
    """返回 (embed_fn, stats)。

    重切后内容变了, 旧向量失效必须重算, 库里的行必须带**真向量** —— 缺向量的行
    在 hybrid / dense 两条路径下都召回不到(见 NULL_CHUNK_RETRIEVAL_WARNING)。

    两种模式:
      - 严格(默认): embed 失败**抛异常**。异常冒泡出 apply_group 的
        session_scope → 该 source_hash 的事务整体回滚, 不留半截 NULL 行;
        再由调用方的 per-group except 计入 unrepaired, 继续下一个文件。
      - 宽松(--allow-null-embedding): 保持旧行为, 失败写 NULL 并只在第一次
        失败时打印原因, 不逐行刷屏、不静默假装成功。调用方需已用预检确认
        失败原因并接受「这些内容暂时检索不到」。
    """
    stats = {"ok": 0, "null": 0}
    state = {"disabled": False}

    def _embed(text: str) -> Optional[List[float]]:
        if state["disabled"]:  # 只有宽松模式会走到这里
            stats["null"] += 1
            return None
        try:
            from services.text_embedding import embed_input

            vector = embed_input(text)
        except Exception as exc:  # noqa: BLE001 - 由调用方按模式决定回滚还是降级
            if not allow_null:
                # 严格模式不置 disabled: 预检已确认 provider 可用, 中途失败多为瞬时,
                # 下一个文件/chunk 应重新尝试而不是整批放弃。
                raise
            state["disabled"] = True
            print(f"  [!] embedding 不可用, 后续 chunk 一律写 NULL: {exc}")
            print(f"  [!] {NULL_CHUNK_RETRIEVAL_WARNING}")
            stats["null"] += 1
            return None
        stats["ok"] += 1
        return vector

    return _embed, stats


def check_embedding_available() -> Optional[str]:
    """embedding 预检: 探针能算出向量返回 None, 否则返回失败原因字符串。

    只探一次, 不落库。目的是在动任何一行数据之前就知道 provider/key 是否可用 ——
    预检过了再中途失败, 由 per-file 事务回滚兜底(P0-2)。
    """
    try:
        from services.text_embedding import embed_input

        vector = embed_input(EMBEDDING_PROBE_TEXT)
    except Exception as exc:  # noqa: BLE001 - 任何失败都算预检不通过
        return str(exc)
    if not vector:
        return "embed_input 返回空向量"
    return None


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def _is_prod_db(db_url: str) -> bool:
    """db_url 是否指向仓库内生产库 data/smart_appointment.db。"""
    from sqlalchemy.engine import make_url

    try:
        database = make_url(db_url).database
    except Exception:  # noqa: BLE001 - 非法 URL 交给后续 create_engine 报错
        return False
    if not database:
        return False
    try:
        return Path(database).resolve() == (PROJECT_ROOT / PROD_DB_RELATIVE).resolve()
    except OSError:
        return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按新分块参数重切存量知识库 chunk(默认 dry-run)",
    )
    parser.add_argument(
        "--db", required=True,
        help="目标库 URL, 如 sqlite:///./data/smart_appointment.db",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="真正写库; 不加则只做 dry-run 输出报告",
    )
    parser.add_argument(
        "--source-dir", default=None,
        help="原始文档目录; 给了就优先直接读原文件重切, 否则用 chunk 拼接还原",
    )
    parser.add_argument(
        "--i-know-this-is-prod", action="store_true", dest="confirm_prod",
        help="--db 指向仓库内 data/smart_appointment.db 时必须显式加此项",
    )
    parser.add_argument(
        "--allow-null-embedding", action="store_true", dest="allow_null_embedding",
        help=(
            "仅对 --apply 有意义: embedding 不可用时仍写 NULL 行。"
            f"默认拒绝({NULL_CHUNK_RETRIEVAL_WARNING})"
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="覆盖分块大小(默认取 CHUNK_SIZE / 500)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help="覆盖分块重叠(默认取 CHUNK_OVERLAP / 50)",
    )
    return parser.parse_args(argv)


def process_group(
    group: Dict[str, Any],
    chunker: TextChunker,
    chunk_overlap: int,
    source_dir: Optional[Path],
) -> Tuple[List[str], str]:
    """还原 + 重切 + 可逆性自检。返回 (new_texts, restored_by)。"""
    full_text: Optional[str] = None
    restored_by = "source_file"
    if source_dir is not None:
        full_text = _read_source_file(source_dir, group["source_file"])
    if full_text is None:
        restored_by = "stitch"
        if group["chunk_index_missing"]:
            raise ValueError("chunk_index 缺失, 无法保证顺序")
        full_text = stitch_chunks(group["contents"], OVERLAP_SEEK_LIMIT)
    if not full_text.strip():
        raise ValueError("还原后为空")

    new_texts = rechunk_exact(chunker, full_text, chunk_overlap)
    # 可逆性自检: 精确切片必须能逐字拼回还原全文, 否则拒绝写库
    if stitch_chunks(new_texts, chunk_overlap) != full_text:
        raise ValueError("重切结果不可逆, 拒绝写库")
    return new_texts, restored_by


def run(args: argparse.Namespace) -> int:
    if _is_prod_db(args.db) and not args.confirm_prod:
        print(
            f"[rechunk] 拒绝执行: --db 指向仓库内生产库 {PROD_DB_RELATIVE}。\n"
            f"[rechunk] 请先复制到临时库, 或确认无误后加 --i-know-this-is-prod。",
            file=sys.stderr,
        )
        return 2

    # embedding 预检: 在写任何一行之前探一次。失败且未显式放行 → 一行都不写。
    # (生产库守卫与 --allow-null-embedding 相互独立, 各自 exit 2 / exit 3。)
    if args.apply and not args.allow_null_embedding:
        reason = check_embedding_available()
        if reason is not None:
            print(
                f"[rechunk] 拒绝写库(exit {EXIT_EMBEDDING_UNAVAILABLE}): "
                f"embedding 预检失败: {reason}\n"
                f"[rechunk] 新块会写成 NULL embedding, 而 {NULL_CHUNK_RETRIEVAL_WARNING}。\n"
                f"[rechunk] 出路: 修好 EMBEDDING_PROVIDER / EMBEDDING_API_KEY 后重跑; "
                f"或确认接受「暂时检索不到」时加 --allow-null-embedding。\n"
                f"[rechunk] 未写入任何行。",
                file=sys.stderr,
            )
            return EXIT_EMBEDDING_UNAVAILABLE

    app_settings = AppSettings()
    chunk_size = args.chunk_size or app_settings.chunk_size
    chunk_overlap = args.chunk_overlap or app_settings.chunk_overlap
    chunker = TextChunker(
        ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    sm = SessionManager(db_path=args.db)
    embed_fn, embed_stats = make_embedder(allow_null=args.allow_null_embedding)
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None

    groups = load_groups(sm)
    total_old = sum(g["old_chunk_count"] for g in groups)
    print(
        f"[rechunk] db={args.db} apply={args.apply} "
        f"chunk_size={chunk_size} overlap={chunk_overlap}"
    )
    print(f"[rechunk] 待处理文件数: {len(groups)} (旧 chunk 共 {total_old} 行)")

    unrepaired: List[str] = []
    written = 0
    # groups 已按 source_file 排序, 所以同一 source_file 的多个 source_hash 相邻;
    # 逐个文件的 try/except 仍保证「一个失败不中断整批」。
    for group in groups:
        label = group["source_file"] or f"<hash:{group['source_hash'][:8]}>"
        try:
            new_texts, restored_by = process_group(
                group, chunker, chunk_overlap, source_dir,
            )
        except Exception as exc:  # noqa: BLE001 - 单文件失败不中断整批
            unrepaired.append(f"{label} ({exc})")
            continue

        digest = hashlib.sha256(
            stitch_chunks(new_texts, chunk_overlap).encode()
        ).hexdigest()[:12]
        print(
            f"  {label}: 旧 chunk 数 {group['old_chunk_count']} → 新 chunk 数 "
            f"{len(new_texts)}  [还原={restored_by} text_sha={digest}]"
        )

        if args.apply:
            try:
                # 严格模式下 embed 失败会在这里抛出, apply_group 的事务整体回滚,
                # 该文件不留 NULL 行; 记入 unrepaired 并继续下一个文件。
                apply_group(sm, group, new_texts, embed_fn)
            except Exception as exc:  # noqa: BLE001 - 单文件失败不中断整批
                reason = str(exc) or exc.__class__.__name__
                unrepaired.append(f"{label} (写库失败已回滚: {reason})")
                print(f"  [!] {label}: 写库失败, 该文件事务已回滚: {reason}")
                continue
            written += 1

    print(
        f"[rechunk] 完成: 文件 {len(groups)} 个, "
        f"{'已写库' if args.apply else 'dry-run 未写库'} {written} 个"
    )
    if args.apply:
        print(
            f"[rechunk] embedding: 成功 {embed_stats['ok']} 块, "
            f"写 NULL {embed_stats['null']} 块"
        )
        if embed_stats["null"]:
            print(
                f"[rechunk] 警告: 有 {embed_stats['null']} 块未算出向量, "
                f"{NULL_CHUNK_RETRIEVAL_WARNING}。"
            )
    if unrepaired:
        print(f"[rechunk] 无法还原的文件 {len(unrepaired)} 个:")
        for item in unrepaired:
            print(f"  - {item}")
    sm.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
