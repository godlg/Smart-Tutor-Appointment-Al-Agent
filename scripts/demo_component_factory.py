"""
组件工厂接入演示: 用 ComponentFactory 创建六类组件 → 组装检索 → 评测。

不替换任何现有调用路径 (knowledge_service / hybrid_retriever 原样保留),
仅演示"可插拔组件层"成立: YAML 里切 type 即可换实现。

用法:
  python scripts/demo_component_factory.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    import asyncio
    import hashlib
    from services.components.base import QueryCase
    from services.components.factory import ComponentFactory
    from services.knowledge_service import KnowledgeService

    # 确定性 fake embedding (演示不依赖远程 API)
    def _fake_embed(text: str):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:16]]

    import services.text_embedding as te
    import services.knowledge_service as ks
    import services.hybrid_retriever as hr
    te.embed_input = _fake_embed
    ks.embed_input = _fake_embed
    hr.embed_input = _fake_embed

    async def _seed() -> KnowledgeService:
        svc = KnowledgeService(db_path="sqlite:///:memory:")
        svc.db.add_document(
            content="高中数学 12 周提分方案: 函数、几何、数列、概率四大模块, 含韦达定理专题。",
            category="course_intro", keywords=["高中", "数学", "高考", "韦达定理"],
            subject="数学", grade_level="高中", doc_type="course_intro",
        )
        svc.db.add_document(
            content="初中数学 一元二次方程: 判别式、根与系数关系(韦达定理)、动点问题。",
            category="subject_method", keywords=["初中", "数学", "一元二次方程", "韦达定理"],
            subject="数学", grade_level="初中", doc_type="subject_method",
        )
        svc.db.add_document(
            content="小学阅读启蒙课: 以经典绘本为载体, 词汇 + 造句 + 复述故事。",
            category="course_intro", keywords=["小学", "阅读", "绘本"],
            subject="语文", grade_level="小学", doc_type="course_intro",
        )
        await svc.initialize()
        return svc

    svc = asyncio.run(_seed())

    cases = [
        QueryCase(query="韦达定理 初中数学", expected_doc_ids=[2]),
        QueryCase(query="小学 绘本 阅读", expected_doc_ids=[3]),
        QueryCase(query="高中数学 提分方案", expected_doc_ids=[1]),
    ]

    factory = ComponentFactory(knowledge_service=svc, evaluator_cases=cases)

    # 1. 六类组件 (LLM/Embedding 只演示对象创建, 不调真实 API)
    llm = factory.get_llm()
    emb = factory.get_embedding()
    splitter = factory.get_splitter()
    vs = factory.get_vector_store()
    reranker = factory.get_reranker()
    evaluator = factory.get_evaluator()

    print(f"llm      = {type(llm).__name__}")
    print(f"embedding= {type(emb).__name__}")
    print(f"splitter = {type(splitter).__name__}  (500/50)")
    print(f"chunks   = {len(splitter.split('高中 数学 12 周提分方案: 函数、几何、数列、概率四大模块。' * 20))} 块")
    print(f"vector   = {type(vs).__name__}")
    print(f"reranker = {type(reranker).__name__}")
    print(f"evaluator= {type(evaluator).__name__}  (k=3)")

    # 2. 真实检索一次 (hybrid 路径)
    results = vs.search("韦达定理 初中数学", top_k=3)
    print("\n-- hybrid 检索 '韦达定理 初中数学' --")
    for r in results:
        print(f"  rank={r.get('rank')} score={r.get('score'):.4f} id={r.get('id')} "
              f"content={str(r.get('content'))[:36]}...")

    # 3. 评测 retriever (Recall@k / MRR)
    metrics = evaluator.evaluate(vs)
    print(f"\n-- Evaluator (Recall@3 / MRR) --")
    print(f"  recall_at_k = {metrics['recall_at_k']:.2f}")
    print(f"  mrr         = {metrics['mrr']:.2f}")
    for case in metrics["case_results"]:
        print(f"  {case['query']!r}: hit={case['hit_ids']} recall={case['recall']:.2f} mrr={case['mrr']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
