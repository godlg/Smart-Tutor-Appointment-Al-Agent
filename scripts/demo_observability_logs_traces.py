"""演示:跑一次真实 hybrid 检索 + 一次 agentic chat,展示 Logs/Traces 输出。"""
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from observability import setup_logging, tracer

setup_logging()  # JSON 行输出


def fake_embed(text):
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


async def demo_hybrid():
    from unittest.mock import MagicMock, patch
    from services.hybrid_retriever import HybridRetriever

    ks = MagicMock()
    ks.db.get_all_documents.return_value = [
        {"id": 1, "content": "高中数学 韦达定理 专题:函数、几何、数列、概率。",
         "category": "subject_method",
         "embedding": fake_embed("高中数学 韦达定理 专题")},
        {"id": 2, "content": "初中数学 一元二次方程 判别式与韦达定理。",
         "category": "subject_method",
         "embedding": fake_embed("初中数学 一元二次方程")},
    ]
    retriever = HybridRetriever(knowledge_service=ks, rrf_k=60, candidates=5, rerank=False)
    with patch("services.hybrid_retriever.embed_input", fake_embed):
        results = await retriever.search("韦达定理怎么学", top_k=2)
    print(f"[demo] hybrid results={len(results)}")


def demo_agentic(tmp_path):
    from db.db_router import DatabaseRouter
    from services.lesson_service import LessonService
    from services.student_profile_service import StudentProfileService
    from agents.agentic.agentic_trial_booking import run_agentic_trial_booking

    db = DatabaseRouter(f"sqlite:///{tmp_path / 'demo.db'}")
    ls, ss = LessonService(), StudentProfileService()
    ls.db, ss.db = db, db
    try:
        ss.initialize_default_student()
    except Exception:
        pass

    resp1 = Mock(tool_calls=[{
        "name": "book_trial_lesson", "id": "c1",
        "args": {"teacher_id": 1, "time_iso": "2026-09-25 19:00", "subject": "数学"},
    }], content="")
    resp2 = Mock(content="已为家长预约张老师 2026-09-25 19:00 的免费试讲!")
    llm = Mock()
    bound = Mock()
    bound.invoke.side_effect = [resp1, resp2]
    llm.bind_tools.return_value = bound

    result = run_agentic_trial_booking(
        llm=llm, teacher_id=1, time_iso="2026-09-25 19:00", subject="数学",
        lesson_service=ls, student_service=ss, user_message="帮我约试讲",
    )
    print(f"[demo] agentic result={result}")
    # Windows: 显式释放 DB 文件句柄,否则 TemporaryDirectory 清理失败
    try:
        db.session_manager.engine.dispose()
    except Exception:
        pass


if __name__ == "__main__":
    import tempfile
    print("=== demo: hybrid search (RAG trace) ===")
    asyncio.run(demo_hybrid())
    print("=== demo: agentic tool loop (tool_call trace) ===")
    with tempfile.TemporaryDirectory() as td:
        demo_agentic(Path(td))
    print("=== done ===")
