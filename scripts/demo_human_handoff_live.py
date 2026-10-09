"""人工接管 P1：真实接管闭环演示脚本

流程:
1. 手搓带 fake history 的 agent(固定返回 unrelated 的 fake classifier + fake router)
2. 连续 3 轮无关输入 → 第 3 轮触发 [HANDOFF] 转人工 → 打印 transcript 快照前 2 条
3. 家长发"大概什么时候能回复我" → 落库 parent 消息
4. 客服 get_pending() 看到记录 → claim(客服小美)
5. 客服 agent_reply("家长您好…已为您预留周六试讲名额")
6. 家长 get_messages_by_session 拉到客服回复并打印
7. resolve → 断言全部通过

用临时 DB,不污染 data/smart_appointment.db。

运行:
    python scripts/demo_human_handoff_live.py
exit 0 = 全部断言通过
"""
import asyncio
import json
import sys
import tempfile
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


class _FakeMessage:
    """模拟 langchain message(type + content)。"""

    def __init__(self, type_, content):
        self.type = type_
        self.content = content


class _FakeHistory:
    def __init__(self, messages):
        self.messages = messages


def _drain(agen):
    async def _run():
        out = []
        async for token in agen:
            out.append(token)
        return "".join(out)
    return asyncio.run(_run())


def _make_agent(handoff_service, session_chat_history):
    """绕过 TaskClassificationAgent.__init__(避免真实 LLM),注入 fake。"""
    from agents.task_classification_agent import TaskClassificationAgent
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    agent = object.__new__(TaskClassificationAgent)
    agent.session_id = "demo-live-session"
    agent.handoff_service = handoff_service
    agent.session_chat_history = session_chat_history
    classifier = _FakeClassifier("unrelated")
    agent.classification_processor = ClassificationProcessor(
        task_classifier=classifier,
        state_manager=StateManager(),
        agent_router=_FakeRouter(),
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=agent.session_id,
        session_chat_history=session_chat_history,
    )
    agent._classifier = classifier
    return agent


def main():
    from db.db_router import DatabaseRouter
    from services.human_handoff_service import HumanHandoffService

    # 临时 DB(不污染 data/smart_appointment.db)
    tmp_dir = tempfile.mkdtemp(prefix="demo_human_handoff_live_")
    db_file = Path(tmp_dir) / "handoff_live.db"
    db_path = f"sqlite:///{db_file.as_posix()}"

    db_router = DatabaseRouter(db_path)
    handoff = HumanHandoffService.__new__(HumanHandoffService)
    handoff.db_router = db_router
    handoff.repo = db_router.handoff_repo
    handoff.threshold = handoff._resolve_threshold(None)
    handoff._kickback_counts = {}
    handoff._handed_off_sessions = set()

    history = _FakeHistory([
        _FakeMessage("human", "我想给高三的孩子找数学老师"),
        _FakeMessage("ai", "好的，方便说下孩子的薄弱点和预算吗？"),
        _FakeMessage("human", "孩子数学函数部分比较弱，预算每课时300左右"),
    ])
    agent = _make_agent(handoff, history)

    print(f"===== 阈值 = {handoff.threshold}，连续无关输入自动转人工 =====")
    unrelated_inputs = ["今天天气怎么样", "给我讲个笑话", "随便聊聊"]
    out = ""
    for i, text in enumerate(unrelated_inputs, start=1):
        out = _drain(agent.classify_task_stream(text))
        handed = "[HANDOFF]" in out
        print(f"  第 {i} 轮 [{text}] → {'转人工 [HANDOFF]' if handed else '能力列表(未转人工)'}")

    assert "[HANDOFF]" in out, "第 3 轮无关输入应触发转人工"
    assert "已为您转接人工顾问老师" in out, "应输出人工触发文案"

    # transcript 快照:会话历史 + 无关轮输入(无关轮不写 history,靠 record_input 补全)
    pending = handoff.get_pending()
    assert len(pending) == 1, "应只建 1 条 pending(幂等)"
    transcript = pending[0]["transcript"]
    assert transcript, "应快照会话历史"
    parsed = json.loads(transcript)
    contents = [item["content"] for item in parsed]
    for text in unrelated_inputs:
        assert text in contents, f"transcript 应含无关轮输入: {text}"
    print(f"\n===== transcript 快照(共 {len(parsed)} 条，末 3 条为无关轮输入)=====")
    for item in parsed:
        mark = "  <- 无关轮" if item["content"] in unrelated_inputs else ""
        print(f"  [{item['role']}] {item['content']}{mark}")

    sid = pending[0]["session_id"]
    hid = pending[0]["id"]

    # 2) 家长发消息 → 落库 parent
    parent_msg = handoff.parent_message(sid, "大概什么时候能回复我")
    print(f"\n===== 家长消息落库 =====")
    print(f"  role={parent_msg['role']} content={parent_msg['content']}")
    assert parent_msg is not None
    assert parent_msg["role"] == "parent"

    # 3) 客服 get_pending 看到记录 → claim
    pendings = handoff.get_pending()
    print(f"\n===== 客服 get_pending 看到 {len(pendings)} 条待处理工单 =====")
    for r in pendings:
        print(f"  id={r['id']} type={r['trigger_type']} status={r['status']} "
              f"agent_id={r['agent_id']}")
    assert len(pendings) == 1

    claimed = handoff.claim(hid, "客服小美")
    print(f"\n===== claim 后 =====")
    print(f"  agent_id={claimed['agent_id']}")
    assert claimed["agent_id"] == "客服小美"

    # 4) 客服回复
    reply = handoff.agent_reply(
        hid, "客服小美",
        "家长您好，我是人工顾问小美，已为您预留周六试讲名额，稍后电话和您确认。",
    )
    print(f"\n===== 客服回复落库 =====")
    print(f"  role={reply['role']} content={reply['content']}")
    assert reply is not None
    assert reply["role"] == "agent"

    # 5) 家长侧轮询拉取客服回复
    session_messages = handoff.get_messages_by_session(sid)
    print(f"\n===== 家长侧 get_messages_by_session 拉到 {len(session_messages)} 条 =====")
    for m in session_messages:
        print(f"  [{m['role']}] {m['content']}")
    assert [m["role"] for m in session_messages] == ["parent", "agent"]
    assert any(
        "已为您预留周六试讲名额" in m["content"] for m in session_messages
    ), "家长应能拉到客服回复"

    # 6) resolve → 恢复 AI 对话
    ok = handoff.resolve(hid, handled_by="客服小美", note="已回电家长")
    record = db_router.handoff_repo.get_by_id(hid)
    print(f"\n===== resolve 后 =====")
    print(f"  status={record['status']} handled_by={record['handled_by']} "
          f"resolved_at={record['resolved_at']}")
    assert ok and record["status"] == "resolved"
    assert handoff.get_pending() == [], "resolve 后不应再有 pending"
    assert handoff.parent_message(sid, "还有人在吗") is None, "resolve 后不再落库"

    db_router.close()
    db_file.unlink(missing_ok=True)
    print("\n演示通过: 快照→家长落库→认领→客服回复→家长拉取→resolve 闭环全部 OK")


if __name__ == "__main__":
    main()
