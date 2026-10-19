"""RAG 评估的真实检索链路装配 (HybridVectorStore + 确定性词袋桩 embedding)。

为什么要这一层 (spec §0):
    v1 的 `_evaluate_rag` 直接用 `Bm25VectorStore`, 只证明「BM25 在 6 篇手写小文上
    能命中」, 门禁绿 ≠ 生产 RAG 可用 —— 它既不过 FAISS dense, 也不过 RRF 融合、
    精排、相关性阈值, 更不碰 query 变换。这里把评估切到生产同款
    `HybridVectorStore` (内部即 `HybridRetriever`), 让门禁测的是真链路。

离线确定性 (词袋桩, spec delta 2 §1):
    embedding 用**确定性词袋桩** (jieba 分词 → 词哈希到固定维度 → 子线性 tf →
    L2 归一化): 词重合越多 cos 越大, 同文本恒同向量, 不依赖远程 embedding API。
    桩 patch 到 `knowledge_service` / `hybrid_retriever` 的 import 点
    (与 tests/test_hybrid_retrieval.py 同款做法)。

    ⚠️ 词袋桩只反映**词重合**, 不代表真实 embedding 的语义召回能力
    (近义改写、跨语言、指代这些它一概不会)。结果里的 `embedding_mode` 因此
    明确标 `"bow_stub"` 而不是 `"stub"` —— 不许把它说成语义模型。
    真实语义数值请用 `EVAL_RAG_LIVE=1` 跑 (真 embedding, 只输出不入门禁)。

    为什么换掉旧的 sha256 桩: 旧桩是**纯噪声** (sha256 → 16 维), dense 路召回
    随机文档、RRF 被随机排名污染, 测出的「hybrid 打不过 BM25」主要是评测工具的
    假象 (spec delta 2 §0-A)。

实时对照:
    `EVAL_RAG_LIVE=1` 时不打桩, 走真实 embedding; 结果只输出、不入门禁。
"""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

# 词袋桩维度 (hashing trick 的桶数)。
# 512 太小: 中文领域词表动辄几千词, 桶数不足会让无关词大量碰撞、稀释词重合的信号;
# 4096 以上对 34 篇语料的区分度不再提升, 只徒增向量体积与计算。
# 2048 在「碰撞可控」与「成本可忽略」之间取平衡 (34 篇 × 2048 维 = 700KB 级)。
STUB_EMBED_DIM = 2048


def _token_bucket(token: str) -> int:
    """token → 桶下标: blake2b 前 8 字节取模。

    用哈希 (hashing trick) 而非全局词表: **函数式确定性**天然成立 ——
    同 token 恒同桶, 与调用顺序、与是否见过其它文本都无关。
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % STUB_EMBED_DIM


def stub_embed(text: str) -> List[float]:
    """确定性**词袋语义桩**: jieba 分词 → 词哈希 → 子线性 tf → L2 归一化。

    - 词重合越多, 两向量 cos 越大 (dense 路因此有真实的"词面相关性"信号);
    - 同文本恒同向量, 与调用顺序无关 (无全局词表, 纯哈希 + 词频累加);
    - 空文本返回全零向量 (与生产 embedding 的空输入行为对齐)。

    与 `services.bm25_index.tokenize` 共用同一分词器: dense 桩与 BM25 路看到的是
    同一套 token, 两条路的差异只来自**加权与融合方式**, 便于归因。
    """
    if not text or not text.strip():
        return [0.0] * STUB_EMBED_DIM

    from services.bm25_index import tokenize

    vector = [0.0] * STUB_EMBED_DIM
    counts: Dict[int, int] = {}
    for token in tokenize(text):
        bucket = _token_bucket(token)
        counts[bucket] = counts.get(bucket, 0) + 1

    # 子线性 tf: 1 + log(tf), 抑制高频词 (长 chunk 里"的/是"这类) 主导向量
    for bucket, count in counts.items():
        vector[bucket] = 1.0 + math.log(count)

    # L2 归一化 → 内积 == 余弦相似度 (FAISS 用的是 IndexFlatIP, 内积检索)
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


def live_mode() -> bool:
    """env EVAL_RAG_LIVE=1/true/yes/on 时走真实 embedding (结果不入门禁)。"""
    return os.getenv("EVAL_RAG_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def patched_embedding() -> Iterator[None]:
    """把确定性桩 patch 到三个 import 点 (text_embedding 源 + 两处 from-import)。

    用 try/finally 手写 patch 而不用 pytest.monkeypatch: 评估器要在 CLI
    (`python -m eval.runner`) 下独立工作, 不该依赖测试夹具。
    """
    if live_mode():
        yield
        return

    from services import hybrid_retriever, knowledge_service, text_embedding

    targets = (
        (text_embedding, "embed_input"),
        (knowledge_service, "embed_input"),
        (hybrid_retriever, "embed_input"),
    )
    saved: List[Tuple[Any, str, Any]] = [
        (mod, attr, getattr(mod, attr)) for mod, attr in targets
    ]
    try:
        for mod, attr, _ in saved:
            setattr(mod, attr, stub_embed)
        yield
    finally:
        for mod, attr, original in saved:
            setattr(mod, attr, original)


def normalized_docs(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把冻结语料规范成检索链路要的 doc 形态 (id/content/keywords/元数据)。

    语料每篇只用 doc_id / content / keywords / category + subject/grade_level/
    doc_type —— 与生产 `knowledge_documents` 行同形, 检索链路拿到的输入与生产一致。
    重复 doc_id 直接报错 (期望标签会因此串味)。
    """
    docs = [
        {
            "id": int(d["doc_id"]),
            "content": d["content"],
            "keywords": list(d.get("keywords", []) or []),
            "category": d.get("category", "general"),
            "subject": d.get("subject"),
            "grade_level": d.get("grade_level"),
            "doc_type": d.get("doc_type"),
        }
        for d in corpus
    ]
    ids = [d["id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"冻结语料 doc_id 重复: {[i for i in ids if ids.count(i) > 1]}")
    return docs


def seed_corpus(ks: Any, docs: List[Dict[str, Any]]) -> None:
    """把语料以**显式 doc_id** 落进 (内存) 知识库并建 FAISS 索引。

    为什么不能用 `KnowledgeService.add_document` / `HybridVectorStore.add`:
        那两个入口走 repo 的自增主键, 会把 corpus 的 doc_id 重排成 1..N。
        而 `HybridRetriever` 的候选集取自 `ks.db.get_all_documents()`,
        检索结果带的就是重排后的 id —— 与 `expected_doc_ids` 完全对不上
        (实测: 期望 79 的 case 返回 id=14), 指标必然全 0。
        ground truth 靠 doc_id 对齐, id 必须原样保真, 所以这里显式写 id。

    仍走 `KnowledgeService.initialize()` 建索引 (公开入口), 只是数据先手工种好。
    """
    import asyncio

    from db.models import KnowledgeDocument

    embeddings = [stub_embed(f"{d['content']} {' '.join(d['keywords'])}".strip()) for d in docs]
    with ks.db.session_manager.session_scope() as session:
        for doc, embedding in zip(docs, embeddings):
            session.add(KnowledgeDocument(
                id=doc["id"],
                content=doc["content"],
                category=doc["category"],
                keywords=doc["keywords"],
                embedding=embedding,
                subject=doc["subject"],
                grade_level=doc["grade_level"],
                doc_type=doc["doc_type"],
                chunk_index=None,
                source_file=None,
                source_hash=None,
                chunk_count=None,
                is_active=1,
            ))
    asyncio.run(ks.initialize())
    if not ks.initialized or ks.index is None:
        raise RuntimeError("冻结语料建索引失败 (KnowledgeService.initialize 未完成)")


def build_corpus_store(corpus: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
    """把冻结 corpus 装进 **HybridVectorStore** (生产同款混合检索器)。

    返回 (store, docs)。store.search 返回的是 HybridRetriever 的结果
    (dense + BM25 + RRF + 精排), 里头的 `id` 即 corpus 的 doc_id —— 与
    `expected_doc_ids` 同一命名空间。
    """
    from services.components.vector_stores import HybridVectorStore
    from services.knowledge_service import KnowledgeService

    docs = normalized_docs(corpus)
    ks = KnowledgeService(db_path="sqlite:///:memory:")
    seed_corpus(ks, docs)
    # HybridVectorStore 内部即 HybridRetriever(ks, ...), 数据已种好, 直接用它检索
    return HybridVectorStore(ks), docs


def query_transform_variants(query: str) -> List[str]:
    """live 模式下把 query 变换成检索串 (含原串, [0] 恒为原串)。

    走 `services.query_transform.variants` 的同步 rule 档 (零 LLM);
    变换失败一律回退 [query], 绝不影响检索。
    """
    try:
        from services.query_transform import variants

        return variants(query) or [query]
    except Exception:  # noqa: BLE001 - 变换失败不影响检索
        return [query]
