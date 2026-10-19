"""RAG 评估集建集工具: 从生产知识库**只读**采样冻结语料 + 生成草稿 cases。

为什么需要它 (spec §0):
    v1 的 `eval/cases/rag.json` 只有 6 篇手写小文 + 8 条逐字改写 query,
    `Recall@3` 恒为 1.0, 门禁毫无区分度。语料必须换成生产库里的真实 chunk,
    query 必须写成「用户真实会问的说法」而不是抄原文。

安全约束 (写死):
    - 生产库一律 `sqlite3.connect("file:<db>?mode=ro", uri=True)` **只读**打开,
      本脚本没有任何写库路径 (不建连接用于写入, 不带 --apply);
    - 建集产物落到 `eval/cases/` 下的**新文件** (rag.corpus.json / rag.draft.json),
      绝不直接覆盖 `rag.json` —— 正式集由人工审阅草稿后落笔。

反逐字改写守卫 (硬性):
    每条 query 与 corpus 任一篇的最长公共子串必须 < 12 字, 否则:
      - 生成阶段: 该候选草稿被丢弃并计入 rejected (脚本继续);
      - `--check` 阶段: 打印违规明细并以 exit 1 失败 (供 CI/测试直接调用)。

【红线】禁止按检索结果回筛用例 (spec delta §1):
    **不允许**「先跑检索探针看哪些 query 能命中, 再把能命中的 query 留下来 / 用它们
    替换现有 case」。那样会把基线刷回接近 1.0, 门禁重新失去区分度 —— 与本次改造
    的目的完全相反。

    定题规则 (唯一合法的流程):
      1. 每条 case 的 `query` 与 `expected_doc_ids` 只由 **`source_quote`**(该 doc 里
         支撑答案的原文片段) 决定, **与检索结果无关**: 先看文档里有什么可答的内容,
         再据此写出用户会怎么问;
      2. 允许替换 case 的理由**只有一条** —— 该 case 的期望 doc 在 corpus 里
         **没有任何可答内容**(`source_quote` 不存在, 或支撑不了 query)。
         这种替换必须在报告里逐条列出:
         `case_id / 原 query / 新 query / 替换理由 / 新 source_quote`;
      3. **禁止**以「检索不到」为由替换或删除任何 case (检索不到正说明它是难例,
         难例是要保留的信号, 不是要消灭的噪声);
      4. case 集合必须保留难例: 最终基线 `recall_at_k` 必须 **<= 0.90**
         (刷成接近满分即视为违规)。

用法:
    ./.venv/Scripts/python.exe scripts/build_rag_eval_cases.py                  # 采样 + 生成草稿
    ./.venv/Scripts/python.exe scripts/build_rag_eval_cases.py --help
    ./.venv/Scripts/python.exe scripts/build_rag_eval_cases.py --check eval/cases/rag.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.rag_guard import (  # noqa: E402
    MIN_COMMON_SUBSTRING,
    corpus_doc_ids_of,
    corpus_texts_of,
    expected_ids_dangling,
    format_violations,
    guard_cases,
    max_common_substring_len,
)

DEFAULT_DB = Path("data") / "smart_appointment.db"
DEFAULT_CORPUS_OUT = PROJECT_ROOT / "eval" / "cases" / "rag.corpus.json"
DEFAULT_DRAFT_OUT = PROJECT_ROOT / "eval" / "cases" / "rag.draft.json"

# 碎块下限: 短于这个字数的 chunk 单独成篇没有检索价值 (spec: 跳过 < 80 字)
MIN_CHUNK_CHARS = 80
# 冻结语料篇数下限 (spec: >= 30)
MIN_CORPUS_DOCS = 30
# 默认采样篇数
DEFAULT_CORPUS_LIMIT = 34

# 查询模板: 口语提问, 刻意避开语料原句 (生成后仍会被守卫逐条复核)
_QUERY_TEMPLATES: Tuple[str, ...] = (
    "孩子这块老是学不明白，该怎么讲",
    "能不能简单说说这是怎么一回事",
    "这里有什么容易搞错的地方吗",
    "这个知识点考得多不多",
    "家长需要提前了解点什么",
)


# ----------------------------------------------------------------------
# 只读采样
# ----------------------------------------------------------------------

def open_readonly(db_path: Path) -> sqlite3.Connection:
    """以只读 URI 打开 SQLite (URI 里的 mode=ro 由 sqlite3 强制, 写操作直接报错)。"""
    resolved = Path(db_path)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    # 双保险: query_only 让误写的 SQL 在 prepare 阶段就失败, 而不是改到磁盘
    conn.execute("PRAGMA query_only = ON")
    return conn


def load_chunks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """读全部活跃 chunk (含溯源 / 元数据列)。"""
    rows = conn.execute(
        "SELECT id, content, category, keywords, subject, grade_level, doc_type, "
        "       source_file, chunk_index "
        "FROM knowledge_documents WHERE is_active = 1"
    ).fetchall()
    chunks: List[Dict[str, Any]] = []
    for (doc_id, content, category, keywords, subject, grade_level,
         doc_type, source_file, chunk_index) in rows:
        chunks.append({
            "doc_id": int(doc_id),
            "content": content or "",
            "category": category,
            "keywords": _parse_keywords(keywords),
            "subject": subject,
            "grade_level": grade_level,
            "doc_type": doc_type,
            "source_file": source_file,
            "chunk_index": chunk_index,
        })
    return chunks


def _parse_keywords(raw: Any) -> List[str]:
    """keywords 列是 TEXT 存 JSON 数组 (SQLAlchemy JSON 列在 sqlite 里即 TEXT)。"""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _normalized(content: str) -> str:
    """去空白后用于内容去重 (与 hybrid_retriever 的去重口径一致)。"""
    return "".join(content.split())


def _stratum_of(chunk: Dict[str, Any]) -> Tuple[str, str, str]:
    """分层键: source_file / subject / grade_level (spec 要求的三维分层)。"""
    return (
        str(chunk.get("source_file") or "<no-source-file>"),
        str(chunk.get("subject") or "<no-subject>"),
        str(chunk.get("grade_level") or "<no-grade>"),
    )


def sample_corpus(
    chunks: Sequence[Dict[str, Any]],
    limit: int = DEFAULT_CORPUS_LIMIT,
    min_chars: int = MIN_CHUNK_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按 source_file / subject / grade_level 分层轮转采样, 返回 (corpus, stats)。

    - 先丢弃 < min_chars 的碎块;
    - 按规范化内容去重 (保留首次出现的);
    - 层内按内容长度降序 (长块信息更完整), 层间轮转取, 保证分层不被单一文件吃满。
    """
    stats = {"total": len(chunks), "too_short": 0, "duplicate": 0, "strata": 0}
    seen: set = set()
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for chunk in chunks:
        content = chunk.get("content") or ""
        if len(content.strip()) < min_chars:
            stats["too_short"] += 1
            continue
        key = _normalized(content)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        buckets.setdefault(_stratum_of(chunk), []).append(chunk)

    stats["strata"] = len(buckets)
    for items in buckets.values():
        items.sort(key=lambda c: (-len(c["content"]), c["doc_id"]))

    ordered: List[Dict[str, Any]] = []
    names = sorted(buckets)
    index = 0
    while len(ordered) < limit and any(buckets[n] for n in names):
        name = names[index % len(names)]
        index += 1
        if buckets[name]:
            ordered.append(buckets[name].pop(0))
    return ordered, stats


def to_corpus_entry(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """冻结语料的一条: {doc_id, content, keywords, subject, grade_level, source_file}。

    额外带上 category / doc_type —— 生产检索链路的候选集必须与 corpus 同形,
    少了这两个字段 BM25 拼接 keywords、category 过滤都会失真。
    """
    return {
        "doc_id": int(chunk["doc_id"]),
        "content": chunk["content"],
        "keywords": list(chunk.get("keywords") or []),
        "subject": chunk.get("subject"),
        "grade_level": chunk.get("grade_level"),
        "source_file": chunk.get("source_file"),
        "category": chunk.get("category", "general"),
        "doc_type": chunk.get("doc_type"),
    }


# ----------------------------------------------------------------------
# 草稿 case 生成
# ----------------------------------------------------------------------

_HEADING_PREFIXES = ("#", "##", "###", "####")
# source_quote 长度上限 (给人工核对用的原文片段, 不宜过长)
QUOTE_MAX_CHARS = 120


def extract_topic(content: str) -> str:
    """从 chunk 里抽一个「话题」短语: 优先 markdown 标题, 退化到首个加粗词。"""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_HEADING_PREFIXES):
            topic = line.lstrip("#").strip()
            # "1.1 机械运动与参照物" → 去掉小节编号
            parts = topic.split(None, 1)
            if len(parts) == 2 and parts[0].rstrip(".").replace(".", "").isdigit():
                topic = parts[1]
            topic = topic.replace("*", "").strip()
            if 2 <= len(topic) <= 30:
                return topic
        if line.startswith("**") and "**" in line[2:]:
            topic = line.split("**")[1].strip()
            if 2 <= len(topic) <= 30:
                return topic
    return ""


def pick_source_quote(content: str, topic: str = "") -> str:
    """挑一句支撑答案的原文片段 (含话题词的那句优先, 否则第一句)。"""
    for raw in content.replace("\n", " ").split("。"):
        sentence = raw.strip().strip("|").strip()
        if len(sentence) < 8:
            continue
        if topic and topic in sentence:
            return sentence[:QUOTE_MAX_CHARS]
    for raw in content.replace("\n", " ").split("。"):
        sentence = raw.strip().strip("|").strip()
        if len(sentence) >= 8:
            return sentence[:QUOTE_MAX_CHARS]
    return content[:QUOTE_MAX_CHARS]


def draft_cases_for(
    corpus: Sequence[Dict[str, Any]],
    corpus_texts: Sequence[str],
    per_doc: int = 1,
    min_verbatim: int = MIN_COMMON_SUBSTRING,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """为冻结语料生成草稿 case (机械起草, 供人工审阅)。

    返回 (drafts, rejected)。rejected 是被反逐字改写守卫拦下的候选 —— 机械模板
    抽出来的话题可能整段落在原文里, 这类候选一律丢弃并上报, 不静默保留。
    """
    drafts: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seq = 0
    for doc in corpus:
        topic = extract_topic(doc["content"])
        if not topic:
            continue
        for tpl in _QUERY_TEMPLATES[:max(0, per_doc)]:
            seq += 1
            query = f"{topic}——{tpl}"
            longest = max_common_substring_len(query, corpus_texts)
            candidate = {
                "id": f"rag_draft_{seq:03d}",
                "query": query,
                "expected_doc_ids": [int(doc["doc_id"])],
                "note": f"机械草稿(话题={topic}), 需人工核对是否符合用户真实问法",
                "source_quote": pick_source_quote(doc["content"], topic),
                "label_source": "from_real_chunk",
            }
            if longest >= min_verbatim:
                rejected.append({**candidate, "max_common": longest})
                continue
            drafts.append(candidate)
    return drafts, rejected


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------

def check_case_file(
    path: Path,
    min_cases: int = 1,
    min_corpus: int = 0,
    min_verbatim: int = MIN_COMMON_SUBSTRING,
) -> Tuple[bool, List[str]]:
    """校验一份 rag 评估集: 反逐字改写 + 悬空期望 + 条数下限。

    返回 (ok, messages)。ok=False 时 messages 是失败原因列表。
    """
    problems: List[str] = []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus = data.get("corpus", [])
    cases = data.get("cases", [])
    corpus_texts = corpus_texts_of(data)

    if len(corpus) < min_corpus:
        problems.append(f"corpus {len(corpus)} 篇 < 要求 {min_corpus} 篇")
    if len(cases) < min_cases:
        problems.append(f"cases {len(cases)} 条 < 要求 {min_cases} 条")

    violations = guard_cases(cases, corpus_texts, min_verbatim)
    if violations:
        problems.append(format_violations(violations))

    dangling = expected_ids_dangling(cases, corpus_doc_ids_of(data))
    if dangling:
        problems.append(
            "悬空期望 (expected_doc_ids 在 corpus 里找不到): "
            + "; ".join(f"{d['id']} → {d['missing']}" for d in dangling)
        )
    return (not problems), problems


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从生产知识库只读采样 RAG 评估语料 + 生成草稿 cases (不写库)",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"生产库路径 (只读打开, 默认 {DEFAULT_DB})")
    parser.add_argument("--corpus-out", default=str(DEFAULT_CORPUS_OUT),
                        help="冻结语料输出路径 (默认 eval/cases/rag.corpus.json)")
    parser.add_argument("--draft-out", default=str(DEFAULT_DRAFT_OUT),
                        help="草稿 case 输出路径 (默认 eval/cases/rag.draft.json)")
    parser.add_argument("--limit", type=int, default=DEFAULT_CORPUS_LIMIT,
                        help=f"冻结语料篇数 (默认 {DEFAULT_CORPUS_LIMIT})")
    parser.add_argument("--min-chunk-chars", type=int, default=MIN_CHUNK_CHARS,
                        help=f"跳过短于该字数的碎块 (默认 {MIN_CHUNK_CHARS})")
    parser.add_argument("--per-doc", type=int, default=1,
                        help="每篇语料起草几条 query (默认 1)")
    parser.add_argument("--check", default=None,
                        help="只校验既有 case 文件 (反逐字改写 + 悬空期望 + 条数), 违规 exit 1")
    parser.add_argument("--check-min-cases", type=int, default=1,
                        help="--check 时的 cases 条数下限 (默认 1)")
    parser.add_argument("--check-min-corpus", type=int, default=0,
                        help="--check 时的 corpus 篇数下限 (默认 0)")
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> int:
    path = Path(args.check)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    ok, problems = check_case_file(
        path, min_cases=args.check_min_cases, min_corpus=args.check_min_corpus,
    )
    print(f"[rag-cases] 校验 {path}")
    if ok:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[rag-cases] OK: corpus={len(data.get('corpus', []))} 篇, "
              f"cases={len(data.get('cases', []))} 条, "
              f"反逐字改写守卫通过 (< {MIN_COMMON_SUBSTRING} 字)")
        return 0
    for problem in problems:
        print(f"[rag-cases] 失败: {problem}", file=sys.stderr)
    return 1


def run_build(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not db_path.exists():
        print(f"[rag-cases] 生产库不存在: {db_path}", file=sys.stderr)
        return 2

    conn = open_readonly(db_path)
    try:
        chunks = load_chunks(conn)
    finally:
        conn.close()

    print(f"[rag-cases] 只读读取 {db_path}: 活跃 chunk {len(chunks)} 条")
    sampled, stats = sample_corpus(chunks, limit=args.limit, min_chars=args.min_chunk_chars)
    corpus = [to_corpus_entry(c) for c in sampled]
    print(
        f"[rag-cases] 采样: {len(corpus)} 篇 (丢弃碎块 {stats['too_short']}, "
        f"重复 {stats['duplicate']}, 分层 {stats['strata']} 组)"
    )
    if len(corpus) < MIN_CORPUS_DOCS:
        print(
            f"[rag-cases] 警告: corpus {len(corpus)} 篇 < spec 要求的 {MIN_CORPUS_DOCS} 篇",
            file=sys.stderr,
        )

    corpus_texts = [c["content"] for c in corpus]
    drafts, rejected = draft_cases_for(
        corpus, corpus_texts, per_doc=args.per_doc,
    )
    print(
        f"[rag-cases] 草稿 {len(drafts)} 条; "
        f"被反逐字改写守卫拦下 {len(rejected)} 条 (query 抄了原文 >= {MIN_COMMON_SUBSTRING} 字)"
    )
    for item in rejected[:5]:
        print(f"  - 丢弃 {item['id']} (公共子串 {item['max_common']} 字): {item['query']!r}")

    corpus_payload = {
        "version": "2.0.0",
        "kind": "rag_corpus",
        "description": "RAG 评估冻结语料: 从生产知识库只读采样 (跳过 <"
                       f"{args.min_chunk_chars} 字碎块, 按内容去重)",
        "source_db": str(db_path.relative_to(PROJECT_ROOT)) if db_path.is_relative_to(PROJECT_ROOT) else str(db_path),
        "corpus": corpus,
    }
    draft_payload = {
        "version": "2.0.0",
        "kind": "rag_draft_cases",
        "description": "RAG 评估草稿 cases: 机械起草, 需人工审阅后并入 rag.json",
        "k": 3,
        "min_common_substring": MIN_COMMON_SUBSTRING,
        "corpus_file": "rag.corpus.json",
        "cases": drafts,
    }

    corpus_out = Path(args.corpus_out)
    draft_out = Path(args.draft_out)
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text(json.dumps(corpus_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    draft_out.write_text(json.dumps(draft_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rag-cases] 已写 {corpus_out}")
    print(f"[rag-cases] 已写 {draft_out}")
    print("[rag-cases] 注意: 本脚本不写 rag.json, 正式集请人工审阅草稿后落笔。")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        return run_check(args)
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
