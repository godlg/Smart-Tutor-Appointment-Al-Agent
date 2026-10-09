"""LLM 分类故障降级演示脚本

对照两组行为(用临时 SQLite):
1. 5 连 LLM 故障(分类器抛 ClassificationUnavailable) → 每次都回「系统繁忙」,
   不踢回、不转人工、状态保持 CLASSIFY(家长稍后可重试)。
2. 3 连真「无关」→ 原踢回计数 → 第 3 轮转人工 [HANDOFF](证明原行为未被破坏)。

运行:
    python scripts/demo_classification_failure.py
exit 0 = 全部断言通过
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _UnavailableClassifier:
    """打桩分类器:模拟 LLM 不可用,固定抛 ClassificationUnavailable。"""

    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        from agents.task_classification.task_classifier import ClassificationUnavailable
        raise ClassificationUnavailable("AccountQuotaExceeded: 429")


class _UnrelatedClassifier:
    """打桩分类器:模拟 LLM 正常返回「无关」。"""

    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        return "unrelated"


class _FakeRouter:
    """带 handle_unsupported_task 的假路由器(不触网)。"""

    def __init__(self):
        self.consultant_agent = None
        self.trial_agent = None
        self.lesson_agent = None
        self.recommendation_agent = None

    def get_available_services(self):
        return []

    async def handle_unsupported_task(self, category):
        yield "[REPLY][归类机器人]这个问题暂时超出我的能力范围。"


def _drain(agen):
    async def _run():
        out = []
        async for token in agen:
            out.append(token)
        return "".join(out)
    return asyncio.run(_run())


def _make_processor(handoff_service, classifier):
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    return ClassificationProcessor(
        task_classifier=classifier,
        state_manager=StateManager(),
        agent_router=_FakeRouter(),
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id="demo-session",
    )


def _make_handoff(db_router):
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService.__new__(HumanHandoffService)
    svc.db_router = db_router
    svc.repo = db_router.handoff_repo
    svc.threshold = svc._resolve_threshold(None)
    svc._kickback_counts = {}
    svc._handed_off_sessions = set()
    svc._recent_inputs = {}
    return svc


def main():
    from db.db_router import DatabaseRouter

    # 临时文件型 SQLite(演示可落库查询),结束即删
    db_file = PROJECT_ROOT / "data" / "_demo_classification_failure.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()
    db_path = f"sqlite:///{db_file.as_posix()}"

    db_router = DatabaseRouter(db_path)
    try:
        # ------------------------------------------------------------------
        # 组一:5 连 LLM 故障 → 不转人工,只回繁忙
        # ------------------------------------------------------------------
        print("===== 组一:5 连 LLM 故障(分类器抛 ClassificationUnavailable) =====")
        handoff = _make_handoff(db_router)
        classifier = _UnavailableClassifier()
        processor = _make_processor(handoff, classifier)

        business_input = "我想给初三的孩子约一节数学试听课"
        for i in range(1, 6):
            out = _drain(processor.process_task_stream(business_input))
            handed = "[HANDOFF]" in out
            print(f"  第 {i} 轮 [正常业务输入] → {'转人工 [HANDOFF]' if handed else out}")
            assert "[HANDOFF]" not in out, f"第 {i} 轮 LLM 故障不应转人工"
            assert out == "[REPLY]系统繁忙，暂时无法理解您的问题，请稍后重试。"

        state = processor.state_manager.get_current_state()
        pending = handoff.get_pending()
        counts = handoff._kickback_counts.get("demo-session", 0)
        print(f"  分类器调用次数 = {classifier.calls}(应为 5)")
        print(f"  踢回计数 = {counts}(应为 0)")
        print(f"  当前状态 = {state.value}(应为 classify)")
        print(f"  DB pending 转人工记录 = {len(pending)} 条(应为 0)")
        assert classifier.calls == 5
        assert counts == 0
        assert state.value == "classify"
        assert pending == []

        # ------------------------------------------------------------------
        # 组二:3 连真无关 → 第 3 轮转人工(原行为) 
        # ------------------------------------------------------------------
        print("\n===== 组二:3 连真「无关」→ 第 3 轮转人工(原行为保持不变) =====")
        handoff2 = _make_handoff(db_router)
        classifier2 = _UnrelatedClassifier()
        processor2 = _make_processor(handoff2, classifier2)

        unrelated_inputs = ["今天天气怎么样", "给我讲个笑话", "随便聊聊"]
        for i, text in enumerate(unrelated_inputs, start=1):
            out = _drain(processor2.process_task_stream(text))
            handed = "[HANDOFF]" in out
            print(f"  第 {i} 轮 [{text}] → {'转人工 [HANDOFF]' if handed else '能力列表(未转人工)'}")

        assert "[HANDOFF]" in out, "第 3 轮真无关应触发转人工"
        assert "已为您转接人工顾问老师" in out
        pending2 = handoff2.get_pending()
        print(f"  第 3 轮完整文案:\n    {out}")
        print(f"  DB pending 记录 = {len(pending2)} 条, "
              f"trigger_type={pending2[0]['trigger_type']}, "
              f"unrelated_count={pending2[0]['unrelated_count']}")
        assert len(pending2) == 1
        assert pending2[0]["trigger_type"] == "kickback_threshold"

        print("\n演示通过: LLM 故障只回繁忙(不踢回/不转人工),真无关仍按阈值转人工")
    finally:
        db_router.close()
        db_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
