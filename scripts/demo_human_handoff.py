"""人工接管（Human Handoff）演示脚本

流程:
1. 手搓固定返回 unrelated 的 fake classifier + fake 下游子 agent
2. 连续 3 轮无关输入 → 第 3 轮触发 [HANDOFF] 转人工
3. 已转人工后再输入"转人工" → 打印短路文案
4. 打印 DB 里 pending 记录 → 调 resolve 标记 → 打印 resolved 后状态

运行:
    python scripts/demo_human_handoff.py
exit 0 = 全部断言通过
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeClassifier:
    """固定返回 unrelated,记录调用次数。"""

    def __init__(self, category="unrelated"):
        self.category = category
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        return self.category


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


def _make_agent(handoff_service):
    """绕过 TaskClassificationAgent.__init__(避免真实 LLM),注入 fake。"""
    from agents.task_classification_agent import TaskClassificationAgent
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    agent = object.__new__(TaskClassificationAgent)
    agent.session_id = "demo-session"
    agent.handoff_service = handoff_service
    classifier = _FakeClassifier("unrelated")
    agent.classification_processor = ClassificationProcessor(
        task_classifier=classifier,
        state_manager=StateManager(),
        agent_router=_FakeRouter(),
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=agent.session_id,
    )
    agent._classifier = classifier
    return agent


def main():
    from db.db_router import DatabaseRouter
    from services.human_handoff_service import HumanHandoffService

    # 用独立临时 SQLite(文件型,演示可落库查)
    db_file = PROJECT_ROOT / "data" / "_demo_human_handoff.db"
    if db_file.exists():
        db_file.unlink()
    db_path = f"sqlite:///{db_file.as_posix()}"

    db_router = DatabaseRouter(db_path)
    handoff = HumanHandoffService.__new__(HumanHandoffService)
    handoff.db_router = db_router
    handoff.repo = db_router.handoff_repo
    handoff.threshold = handoff._resolve_threshold(None)
    handoff._kickback_counts = {}
    handoff._handed_off_sessions = set()

    agent = _make_agent(handoff)

    print(f"===== 阈值 = {handoff.threshold}，连续无关输入自动转人工 =====")
    unrelated_inputs = ["今天天气怎么样", "给我讲个笑话", "随便聊聊"]
    for i, text in enumerate(unrelated_inputs, start=1):
        out = _drain(agent.classify_task_stream(text))
        handed = "[HANDOFF]" in out
        print(f"  第 {i} 轮 [{text}] → {'转人工 [HANDOFF]' if handed else '能力列表(未转人工)'}")

    # 第 3 轮(阈值=3)必须触发
    assert "[HANDOFF]" in out, "第 3 轮无关输入应触发转人工"
    assert "已为您转接人工顾问老师" in out, "应输出人工触发文案"
    print(f"\n  第 3 轮完整文案:\n    {out}")

    # 2) 已转人工后再输入"转人工" → 短路文案(分类器零调用)
    calls_before = agent._classifier.calls
    short_circuit = _drain(agent.classify_task_stream("转人工"))
    print(f"\n===== 已转人工后再输入『转人工』→ 短路文案 =====")
    print(f"    {short_circuit}")
    assert "您的会话已转人工顾问处理中" in short_circuit, "应输出已转人工短路文案"
    assert agent._classifier.calls == calls_before, "已转人工后分类器不应再被调用"

    # 3) DB pending 记录
    pending = handoff.get_pending()
    print(f"\n===== DB pending 记录 {len(pending)} 条 =====")
    for r in pending:
        print(f"    id={r['id']} type={r['trigger_type']} count={r['unrelated_count']} "
              f"status={r['status']}")
    assert len(pending) == 1, "应只建 1 条 pending(幂等)"
    assert pending[0]["trigger_type"] == "kickback_threshold"
    assert pending[0]["unrelated_count"] >= handoff.threshold

    # 4) resolve 标记 → 打印 resolved 后状态
    ok = handoff.resolve(pending[0]["id"], handled_by="客服小美", note="已回电家长")
    assert ok, "resolve 应成功"
    record = db_router.handoff_repo.get_by_id(pending[0]["id"])
    print(f"\n===== resolve 后 =====")
    print(f"    status={record['status']} handled_by={record['handled_by']} "
          f"resolved_at={record['resolved_at']}")
    assert record["status"] == "resolved"
    assert record["handled_by"] == "客服小美"
    assert record["resolved_at"] is not None
    assert handoff.get_pending() == [], "resolve 后不应再有 pending"

    db_router.close()
    db_file.unlink(missing_ok=True)
    print("\n演示通过: 阈值自动转人工 + 已转人工短路 + 落库/处理 全部 OK")


if __name__ == "__main__":
    main()
