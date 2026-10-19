"""RAG pipeline 加固可实跑演示(临时 sqlite + 确定性 embedding 桩, 不碰生产库)。

打印三组证据:
1. 三级降级: embed 正常 → BM25 正常对比; 人为让 dense 失败 → 仍返回 BM25 命中(条数)
2. 阈值: 同一批候选在「阈值关闭 / 阈值 0.0」下的结果条数对比
3. prompt 引用: 构造 2 条知识, 打印 _build_knowledge_context 的真实输出

用法:
  ./.venv/Scripts/python.exe scripts/demo_rag_pipeline_hardening.py
退出码 0 = 全部断言通过。
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("RERANK_ENABLED", "false")  # 演示不走真实精排模型


def _fake_embed(text: str) -> List[float]:
    """sha256 → 16 维确定性向量, 不联网。"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


def _install_embed_stub() -> None:
    from services import text_embedding, knowledge_service, hybrid_retriever

    text_embedding.embed_input = _fake_embed
    knowledge_service.embed_input = _fake_embed
    hybrid_retriever.embed_input = _fake_embed


def _build_corpus() -> "object":
    """临时内存库 + 3 条种子(不写生产库)。"""
    from services.knowledge_service import KnowledgeService

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    svc.db.add_document(
        content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
        category="subject_method", keywords=["初中", "数学", "韦达定理"],
        subject="数学", grade_level="初中", doc_type="subject_method",
    )
    svc.db.add_document(
        content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
        category="course_intro", keywords=["小学", "阅读", "绘本"],
        subject="语文", grade_level="小学", doc_type="course_intro",
    )
    svc.db.add_document(
        content="如何帮孩子选家教老师: 看学科匹配、年级段、教学风格, 建议先试讲。",
        category="parent_guide", keywords=["家长", "老师", "试讲"],
        subject=None, grade_level=None, doc_type="parent_guide",
    )
    asyncio.run(svc.initialize())
    return svc


# ----------------------------------------------------------------------
# 1. 三级降级
# ----------------------------------------------------------------------

def demo_degradation(svc) -> bool:
    from services import hybrid_retriever as hr_mod
    from services.hybrid_retriever import HybridRetriever
    from observability import trace_span

    print("\n=== 1. 三级降级: dense+BM25 → BM25-only → 空 ===")
    retriever = HybridRetriever(svc, rrf_k=60, candidates=20, rerank=False)
    query = "韦达定理 一元二次方程 初中数学"

    with trace_span("demo_healthy") as span:
        healthy = asyncio.run(retriever._search_impl(
            query=query, top_k=3, category=None, subject=None,
            grade_level=None, doc_type=None, span=span,
        ))
    print(f"  [正常] dense + BM25 双路: 命中 {len(healthy)} 条, degraded={span.attrs.get('degraded')}")

    original = hr_mod.embed_input

    def _boom(_q):
        raise RuntimeError("模拟 embedding API 429 / 超时")

    hr_mod.embed_input = _boom
    try:
        with trace_span("demo_degraded") as span2:
            degraded = asyncio.run(retriever._search_impl(
                query=query, top_k=3, category=None, subject=None,
                grade_level=None, doc_type=None, span=span2,
            ))
    finally:
        hr_mod.embed_input = original

    print(f"  [dense 失败] BM25-only: 命中 {len(degraded)} 条, "
          f"degraded={span2.attrs.get('degraded')}")
    if degraded:
        print(f"               top1 = {degraded[0]['content'][:40]}... "
              f"(rrf_score={degraded[0].get('rrf_score', 0):.5f})")

    hr_mod.embed_input = _boom
    try:
        with trace_span("demo_empty") as span3:
            empty = asyncio.run(retriever._search_impl(
                query="量子纠缠 弦论 超导 虫洞", top_k=3, category=None,
                subject=None, grade_level=None, doc_type=None, span=span3,
            ))
    finally:
        hr_mod.embed_input = original
    print(f"  [两路皆空] 返回 {len(empty)} 条, degraded={span3.attrs.get('degraded')} "
          f"(不抛异常, 交给上层降级文案)")

    ok = bool(healthy) and bool(degraded) and empty == []
    print(f"  => 断言: 正常非空({bool(healthy)}) / "
          f"dense 失败仍非空({bool(degraded)}) / 两路皆空返回 []({empty == []}) "
          f"→ {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------
# 2. 阈值
# ----------------------------------------------------------------------

class _StubReranker:
    """按 content 首词给固定分: 5.0 / -3.0 / 4.0(负分应被阈值 0.0 丢弃)。"""

    SCORES: Dict[str, float] = {"AAAA": 5.0, "BBBB": -3.0, "CCCC": 4.0}

    def __init__(self, module) -> None:
        self._module = module
        self._old_avail = module.reranker.available
        self._old_rerank = module.reranker.rerank

    def __enter__(self):
        self._module.reranker.available = lambda: True
        self._module.reranker.rerank = self._rerank
        return self

    def __exit__(self, *exc):
        self._module.reranker.available = self._old_avail
        self._module.reranker.rerank = self._old_rerank
        return False

    def _rerank(self, _query, docs, top_k=None):
        out = []
        for doc in docs:
            marker = doc["content"].split()[0]
            if marker not in self.SCORES:
                continue
            item = dict(doc)
            item["rerank_score"] = self.SCORES[marker]
            out.append(item)
        out.sort(key=lambda d: d["rerank_score"], reverse=True)
        return out[:top_k] if top_k is not None else out


def demo_threshold() -> bool:
    from services.knowledge_service import KnowledgeService
    from services.hybrid_retriever import HybridRetriever
    from services import hybrid_retriever as hr_mod

    print("\n=== 2. 精排阈值: 阈值关闭 vs 阈值 0.0 ===")
    svc = KnowledgeService(db_path="sqlite:///:memory:")
    svc.db.add_document(content="AAAA 初中数学 韦达定理 判别式", category="c", keywords=[])
    svc.db.add_document(content="BBBB 小学英语 自然拼读 常见词", category="c", keywords=[])
    svc.db.add_document(content="CCCC 初中数学 根与系数关系 韦达定理", category="c", keywords=[])
    asyncio.run(svc.initialize())

    query = "韦达定理 初中数学"
    with _StubReranker(hr_mod):
        off = HybridRetriever(svc, candidates=20, rerank=True, min_rerank_score=-99.0)
        kept_off = asyncio.run(off.search(query, top_k=5))

        on = HybridRetriever(svc, candidates=20, rerank=True, min_rerank_score=0.0)
        kept_on = asyncio.run(on.search(query, top_k=5))

    print(f"  阈值关闭(min=-99.0): {len(kept_off)} 条 → "
          f"{[(d['content'].split()[0], round(d['score'], 2)) for d in kept_off]}")
    print(f"  阈值 0.0          : {len(kept_on)} 条 → "
          f"{[(d['content'].split()[0], round(d['score'], 2)) for d in kept_on]}")

    # RRF 兜底路径不应用阈值
    rrf_only = HybridRetriever(svc, candidates=20, rerank=False, min_rerank_score=999.0)
    kept_rrf = asyncio.run(rrf_only.search(query, top_k=5))
    print(f"  RRF 兜底(min=999.0, rerank=False): {len(kept_rrf)} 条 "
          f"(阈值不生效, 否则会被全过滤)")

    ok = (
        len(kept_off) == 3
        and len(kept_on) == 2
        and all(d["score"] >= 0.0 for d in kept_on)
        and bool(kept_rrf)
    )
    print(f"  => 断言: 关闭 3 条({len(kept_off)}) / 阈值 0.0 丢弃负分只剩 2 条"
          f"({len(kept_on)}) / RRF 兜底不受阈值影响({len(kept_rrf)} 条) "
          f"→ {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------
# 3. prompt 来源引用
# ----------------------------------------------------------------------

def demo_prompt_citations() -> bool:
    from agents.consultant.prompt_builder import PromptBuilder

    print("\n=== 3. prompt 来源引用 ===")
    docs = [
        {
            "content": "初中数学一元二次方程的核心是判别式与根与系数关系(韦达定理)。",
            "category": "subject_method",
        },
        {
            "content": "小学阅读启蒙课用经典绘本做载体, 训练词汇、造句与复述。",
            "category": "course_intro",
        },
    ]
    text = PromptBuilder().build_knowledge_context(docs)
    print("  --- _build_knowledge_context 真实输出 ---")
    for line in text.strip().splitlines():
        print(f"  | {line}")
    print("  --- 空输入的兜底文案 ---")
    empty_text = PromptBuilder().build_knowledge_context([])
    print(f"  | {empty_text}")

    ok = (
        "[资料1|subject_method]" in text
        and "[资料2|course_intro]" in text
        and empty_text
        == "没有找到直接相关的知识库信息，请基于教育学常识回答，并标明「仅供参考」。"
    )
    print(f"  => 断言: 含 [资料N|分类] 编号且空输入保留原兜底文案 → "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    _install_embed_stub()
    print("=" * 68)
    print("RAG pipeline 加固演示 (临时内存库 + 确定性 embedding 桩, 不碰生产库)")
    print("=" * 68)

    results = [
        demo_degradation(_build_corpus()),
        demo_threshold(),
        demo_prompt_citations(),
    ]

    print("\n" + "=" * 68)
    passed = sum(1 for r in results if r)
    print(f"结果: {passed}/{len(results)} 组证据通过")
    print("=" * 68)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
