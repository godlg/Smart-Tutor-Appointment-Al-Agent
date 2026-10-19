"""Query 变换(多查询检索 + 改写)可实跑演示(临时内存库 + 确定性 embedding 桩)。

打印同一批 5 条口语化 query 在 off / rule / llm 三档下:
1. 每条的**变体列表**(原始 query 恒在 [0]);
2. 该档 top-3 命中的 doc_id / score(体现多路召回带来的差异)。

llm 档用**确定性桩 LLM**(本演示离线, 不联网、不调用真实模型);另附一例桩 LLM
超时 → 自动回退 rule 档的证据。

用法:
  ./.venv/Scripts/python.exe scripts/demo_query_transform.py
退出码 0 = 全部断言通过。
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("RERANK_ENABLED", "false")  # 演示不走真实精排模型

# 同一批口语化 query: 每条都覆盖一类规则(术语归一 / 指代补全 / 无命中)
QUERIES: List[Dict[str, str]] = [
    {"query": "试听一节课多少钱", "context": "用户: 孩子高二,数学想提分"},
    {"query": "这个怎么样", "context": "用户: 初二数学跟不上,想补课"},
    {"query": "1对1补课贵不贵", "context": "用户: 高一英语"},
    {"query": "网课和线下课哪个好", "context": "用户: 初三物理"},
    {"query": "高中的课程安排是怎样的", "context": "用户: 随便问问"},
]


def _fake_embed(text: str) -> List[float]:
    """sha256 → 16 维确定性向量, 不联网。"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


def _install_embed_stub() -> None:
    from services import text_embedding, knowledge_service, hybrid_retriever

    text_embedding.embed_input = _fake_embed
    knowledge_service.embed_input = _fake_embed
    hybrid_retriever.embed_input = _fake_embed


def _build_corpus():
    """临时内存库 + 6 条种子(不写生产库 / data/smart_appointment.db)。"""
    from services.knowledge_service import KnowledgeService

    svc = KnowledgeService(db_path="sqlite:///:memory:")
    seeds = [
        ("初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
         ["初中", "数学", "韦达定理"], "数学", "初中", "subject_method"),
        ("高中数学 12 周提分方案: 函数、几何、数列、概率四大模块, 含一对一辅导与错题梳理。",
         ["高中", "数学", "高考", "一对一"], "数学", "高中", "course_intro"),
        ("高一英语提分课程: 语法体系 + 长难句拆解 + 高考真题精讲, 含收费标准说明。",
         ["高中", "英语", "语法"], "英语", "高中", "course_intro"),
        ("初三物理补习方案: 力学、电学专题突破, 一对一授课, 价格按课时计。",
         ["初中", "物理", "一对一"], "物理", "初中", "course_intro"),
        ("初二数学学习困难干预: 计算薄弱专项训练, 每周一次阶段测试。",
         ["初中", "数学", "计算"], "数学", "初中", "subject_method"),
        ("授课形式说明: 支持线上课与线下课两种形式, 均可安排试讲;收费标准按课时段区分。",
         ["授课形式", "线上", "线下", "试讲", "收费标准"],
         None, None, "parent_guide"),
    ]
    for content, keywords, subject, grade, doc_type in seeds:
        svc.db.add_document(
            content=content, category=doc_type, keywords=keywords,
            subject=subject, grade_level=grade, doc_type=doc_type,
        )
    asyncio.run(svc.initialize())
    return svc


class _StubLLM:
    """确定性桩 LLM: 按原 query 返回固定的 2 条改写, 不联网。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, prompt: str):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        raw = prompt.rsplit("用户问题:", 1)[-1].strip()
        if "多少钱" in raw or "贵不贵" in raw:
            reply = '["课程 收费标准 价格 课时", "试讲 费用 报价"]'
        elif "网课" in raw or "线下课" in raw:
            reply = '["授课形式 线上课 线下课 对比", "试讲 授课形式 选择建议"]'
        else:
            reply = '["课程 教学安排 阶段测试", "一对一 辅导 提分方案"]'

        class _Msg:
            content = reply

        return _Msg()


def _collect_variants(svc, mode: str, retriever, llm=None):
    """按 mode 收集每条 query 的变体列表 + top-3 命中。"""
    from services.query_transform import variants, variants_async

    rows = []
    for item in QUERIES:
        query, context = item["query"], item["context"]
        if mode == "llm":
            variants_list = asyncio.run(variants_async(
                query, context_text=context, history_text=context,
                mode="llm", llm=llm, timeout=1.0,
            ))
        else:
            variants_list = variants(
                query, context_text=context, history_text=context, mode=mode,
            )
        hits = asyncio.run(retriever.search_multi(variants_list, top_k=3))
        rows.append((query, variants_list, hits))
    return rows


def _print_rows(mode: str, rows) -> None:
    print(f"\n--- 档位 {mode} ---")
    for query, variants_list, hits in rows:
        print(f"  query={query!r}  variants={len(variants_list)}")
        for i, variant in enumerate(variants_list):
            tag = "原始" if i == 0 else f"变体{i}"
            print(f"      [{tag}] {variant}")
        if hits:
            rendered = [
                f"doc{h['id']}({h['score']:.5f})" for h in hits
            ]
            print(f"      top{len(hits)}: {', '.join(rendered)}")
            for h in hits:
                print(f"        - doc{h['id']} score={h['score']:.5f} "
                      f"{h['content'][:34]}...")
        else:
            print("      top0: (无命中)")


def main() -> int:
    from services.hybrid_retriever import HybridRetriever

    _install_embed_stub()
    print("=" * 72)
    print("Query 变换演示 (临时内存库 + 确定性 embedding 桩, 不碰生产库)")
    print("=" * 72)

    svc = _build_corpus()
    retriever = HybridRetriever(svc, rrf_k=60, candidates=20, rerank=False)

    llm = _StubLLM()
    results = {}
    for mode in ("off", "rule", "llm"):
        rows = _collect_variants(svc, mode, retriever, llm=llm)
        results[mode] = rows
        _print_rows(mode, rows)

    # ---- 契约断言 ----
    print("\n--- 契约断言 ---")
    ok = True
    for mode in ("off", "rule", "llm"):
        for query, variants_list, _hits in results[mode]:
            nonempty = bool(variants_list)
            first = variants_list[0] == query
            unique = len(variants_list) == len(set(variants_list))
            capped = len(variants_list) <= 3
            good = nonempty and first and unique and capped
            ok = ok and good
            if not good:
                print(f"  FAIL {mode} query={query!r} → {variants_list}")
    print(f"  off 档全部 == [query]: "
          f"{all(v == [q] for q, v, _ in results['off'])}")
    print(f"  rule 档产生变体条数: "
          f"{sum(len(v) - 1 for _, v, _ in results['rule'])}")
    print(f"  llm  档产生变体条数: "
          f"{sum(len(v) - 1 for _, v, _ in results['llm'])} "
          f"(含 rule 变体, 桩 LLM 调用 {llm.calls} 次)")
    print(f"  契约(非空 / [0]==原串 / 去重 / <=3): {'PASS' if ok else 'FAIL'}")

    # ---- LLM 超时回退证据 ----
    from services.query_transform import variants, variants_async

    slow = _StubLLM(delay=0.5)
    fallback = asyncio.run(variants_async(
        "试听一节课多少钱", mode="llm", llm=slow, timeout=0.05,
    ))
    rule_expected = variants("试听一节课多少钱", mode="rule")
    timeout_ok = fallback == rule_expected and fallback[0] == "试听一节课多少钱"
    print(f"\n--- LLM 超时回退(timeout=0.05s, 桩 sleep 0.5s)---")
    print(f"  回退结果: {fallback}")
    print(f"  == rule 档结果: {timeout_ok} → {'PASS' if timeout_ok else 'FAIL'}")

    # ---- off / rule / llm 的召回差异 ----
    print("\n--- 召回差异(同一 top_k=3)---")
    diff_found = False
    for (q, _v_off, off_hits), (_q2, _v_rule, rule_hits), (_q3, _v_llm, llm_hits) in zip(
        results["off"], results["rule"], results["llm"]
    ):
        off_ids = [h["id"] for h in off_hits]
        rule_ids = [h["id"] for h in rule_hits]
        llm_ids = [h["id"] for h in llm_hits]
        if off_ids != rule_ids or rule_ids != llm_ids:
            diff_found = True
            print(f"  {q!r}: off={off_ids} → rule={rule_ids} → llm={llm_ids}")

    # ---- 全量回归脚本(无外部依赖) ----
    print("\n" + "=" * 72)
    verdict = ok and timeout_ok and bool(diff_found)
    print(f"结果: 契约 {'PASS' if ok else 'FAIL'} / 超时回退 "
          f"{'PASS' if timeout_ok else 'FAIL'} / 三档召回有差异={diff_found}")
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
