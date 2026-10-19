"""
人工接管 P1：真实接管闭环测试

覆盖需求:
- transcript 快照:keyword / threshold 触发都把会话历史快照进 DB
- claim:认领 agent_id,重复认领幂等
- agent_reply:pending 落库 role='agent';resolve 后返回 None
- parent_message:已转人工落库 role='parent';未转人工返回 None
- 已转人工后 classify_task_stream 仍回原固定文案(回归)+ DB 新增 parent 消息
- 客服 API 端到端:reply → GET messages 升序含 parent+agent;GET session messages 拉到回复
- resolve 后 parent_message 返回 None

约束:无 pytest-asyncio,异步用例用 _run(_collect) 同步包装;不触网(手搓 fake);
不硬编码日期;DB 隔离(每用例独立 in-memory 库)。
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


def _login_console(client, tmp_path, monkeypatch, username="admin",
                   password="admin12345", filename="console_live.db"):
    """客服侧 API 已挂控制台闸门:显式临时库播种默认账号并登录拿 cookie。

    显式传临时库路径,不依赖 ConsoleAuthService 的默认(生产)库。
    """
    from api.core import console_auth as console_dep
    from config.console_auth import CONSOLE_COOKIE_NAME
    from services.console_auth_service import ConsoleAuthService

    console_db = f"sqlite:///{(tmp_path / filename).as_posix()}"
    original_init = ConsoleAuthService.__init__

    def _patched_init(self, db_path=None, *args, **kwargs):
        original_init(self, db_path=db_path or console_db)

    monkeypatch.setattr(ConsoleAuthService, "__init__", _patched_init)
    monkeypatch.setattr(console_dep, "_console_auth_service", None, raising=False)
    ConsoleAuthService(db_path=console_db).ensure_default_users()

    response = client.post("/api/console/login",
                           json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    assert CONSOLE_COOKIE_NAME in response.cookies


HANDOFF_REPLY = (
    "抱歉，连续几次没能帮您解决问题，已为您转接人工顾问老师。"
    "您的对话记录已保存，顾问老师会尽快与您联系。"
)
HANDED_OFF_REPLY = (
    "您的会话已转人工顾问处理中，请稍候。如需加急可拨打前台电话咨询。"
)


# ----------------------------------------------------------------------
# 测试替身（不触网）
# ----------------------------------------------------------------------

class _FakeClassifier:
    """固定返回指定分类结果,并记录被调次数。"""

    def __init__(self, category="unrelated"):
        self.category = category
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        return self.category


class _FakeRouter:
    """带 handle_unsupported_task 的假路由器(无需真实子 agent)。"""

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


def _make_history(messages):
    """返回一个带 .messages 属性的假 history 对象。"""
    class _FakeHistory:
        def __init__(self, msgs):
            self.messages = msgs

    return _FakeHistory(messages)


def _make_handoff_service(db_router, threshold=None):
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService.__new__(HumanHandoffService)
    svc.db_router = db_router
    svc.repo = db_router.handoff_repo
    svc.threshold = svc._resolve_threshold(threshold)
    svc._kickback_counts = {}
    svc._handed_off_sessions = set()
    return svc


@pytest.fixture
def db_router():
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


@pytest.fixture
def handoff_service(db_router):
    return _make_handoff_service(db_router)


def _make_processor(handoff_service, classifier=None, router=None,
                    session_id="test-session", session_chat_history=None):
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    classifier = classifier or _FakeClassifier("unrelated")
    router = router or _FakeRouter()
    return ClassificationProcessor(
        task_classifier=classifier,
        state_manager=StateManager(),
        agent_router=router,
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=session_id,
        session_chat_history=session_chat_history,
    )


def _make_agent(handoff_service, classifier=None, router=None,
                session_id="test-session", session_chat_history=None):
    """手搓 TaskClassificationAgent:注入 fake,不触网。"""
    from agents.task_classification_agent import TaskClassificationAgent

    agent = object.__new__(TaskClassificationAgent)
    agent.session_id = session_id
    agent.handoff_service = handoff_service
    agent.session_chat_history = session_chat_history
    processor = agent.classification_processor = _make_processor(
        handoff_service, classifier=classifier, router=router,
        session_id=session_id, session_chat_history=session_chat_history,
    )
    agent._fake_classifier = processor.task_classifier
    agent._fake_router = processor.agent_router
    return agent


# ----------------------------------------------------------------------
# 1. transcript 快照
# ----------------------------------------------------------------------

class TestTranscriptSnapshot:

    def test_snapshot_transcript_formats_json(self, handoff_service):
        history = _make_history([
            _FakeMessage("human", "我想给高三的孩子找数学老师"),
            _FakeMessage("ai", "好的，方便说下预算吗"),
        ])
        snap = handoff_service.snapshot_transcript(history.messages)
        assert snap is not None
        parsed = json.loads(snap)
        assert parsed == [
            {"role": "human", "content": "我想给高三的孩子找数学老师"},
            {"role": "ai", "content": "好的，方便说下预算吗"},
        ]

    def test_snapshot_empty_returns_none(self, handoff_service):
        assert handoff_service.snapshot_transcript([]) is None
        assert handoff_service.snapshot_transcript(None) is None

    def test_snapshot_unknown_message_falls_back(self, handoff_service):
        class _Weird:
            content = 123

        snap = handoff_service.snapshot_transcript([_Weird()])
        parsed = json.loads(snap)
        assert parsed[0]["role"] == "unknown"
        assert parsed[0]["content"] == "123"

    def test_keyword_trigger_snapshots_transcript(self, handoff_service, db_router):
        history = _make_history([_FakeMessage("human", "我要转人工")])
        agent = _make_agent(handoff_service, session_chat_history=history)
        _collect(agent.classify_task_stream("转人工"))
        record = db_router.handoff_repo.get_pending()[0]
        assert record["transcript"] is not None
        parsed = json.loads(record["transcript"])
        assert parsed[0]["content"] == "我要转人工"
        # 触发轮输入也被 record_input 记入缓冲并补进快照
        assert [item["content"] for item in parsed] == ["我要转人工", "转人工"]

    def test_threshold_trigger_snapshots_transcript(self, handoff_service, db_router):
        history = _make_history([
            _FakeMessage("human", "今天天气怎么样"),
            _FakeMessage("ai", "这个问题暂时超出我的能力范围。"),
        ])
        agent = _make_agent(
            handoff_service,
            classifier=_FakeClassifier("unrelated"),
            session_chat_history=history,
        )
        _collect(agent.classify_task_stream("无关一"))
        _collect(agent.classify_task_stream("无关二"))
        out = _collect(agent.classify_task_stream("无关三"))
        assert "[HANDOFF]" in out
        record = db_router.handoff_repo.get_pending()[0]
        assert record["trigger_type"] == "kickback_threshold"
        assert record["transcript"] is not None
        parsed = json.loads(record["transcript"])
        # 2 条历史 + 3 条无关轮输入(无关轮不写历史,靠缓冲补全)
        assert [item["content"] for item in parsed] == [
            "今天天气怎么样", "这个问题暂时超出我的能力范围。",
            "无关一", "无关二", "无关三",
        ]

    def test_no_history_transcript_uses_recent_input(self, handoff_service, db_router):
        # 无会话历史时,快照退化为缓冲里的最近用户输入(不再为空)
        agent = _make_agent(handoff_service, session_chat_history=None)
        _collect(agent.classify_task_stream("转人工"))
        record = db_router.handoff_repo.get_pending()[0]
        assert json.loads(record["transcript"]) == [
            {"role": "human", "content": "转人工"}
        ]


# ----------------------------------------------------------------------
# 2. claim
# ----------------------------------------------------------------------

class TestClaim:

    def test_claim_sets_agent_id(self, handoff_service, db_router):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        claimed = handoff_service.claim(record["id"], "客服小美")
        assert claimed is not None
        assert claimed["agent_id"] == "客服小美"
        assert db_router.handoff_repo.get_by_id(record["id"])["agent_id"] == "客服小美"

    def test_claim_idempotent_keeps_first_agent(self, handoff_service):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        first = handoff_service.claim(record["id"], "客服小美")
        second = handoff_service.claim(record["id"], "客服老王")
        assert first["agent_id"] == second["agent_id"] == "客服小美"

    def test_claim_unknown_returns_none(self, handoff_service):
        assert handoff_service.claim(999999, "客服小美") is None


# ----------------------------------------------------------------------
# 3. agent_reply
# ----------------------------------------------------------------------

class TestAgentReply:

    def test_reply_when_pending(self, handoff_service, db_router):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        message = handoff_service.agent_reply(record["id"], "客服小美", "您好，我是小美")
        assert message is not None
        assert message["role"] == "agent"
        assert message["content"] == "您好，我是小美"
        msgs = db_router.handoff_repo.get_messages(record["id"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "agent"

    def test_reply_after_resolve_returns_none(self, handoff_service, db_router):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        handoff_service.resolve(record["id"], handled_by="客服小美")
        assert handoff_service.agent_reply(record["id"], "客服小美", "晚了") is None
        assert db_router.handoff_repo.get_messages(record["id"]) == []

    def test_reply_unknown_returns_none(self, handoff_service):
        assert handoff_service.agent_reply(999999, "客服小美", "hi") is None


# ----------------------------------------------------------------------
# 4. parent_message
# ----------------------------------------------------------------------

class TestParentMessage:

    def test_parent_message_when_handed_off(self, handoff_service, db_router):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        message = handoff_service.parent_message("s1", "大概什么时候能回复我")
        assert message is not None
        assert message["role"] == "parent"
        assert message["handoff_id"] == record["id"]
        assert message["content"] == "大概什么时候能回复我"

    def test_parent_message_without_handoff_returns_none(self, handoff_service, db_router):
        assert handoff_service.parent_message("no-such-session", "在吗") is None
        assert db_router.handoff_repo.get_messages_by_session("no-such-session") == []

    def test_parent_message_after_resolve_returns_none(self, handoff_service):
        record = handoff_service.trigger_handoff("s1", "keyword", reason="人工")
        handoff_service.resolve(record["id"], handled_by="客服小美")
        assert handoff_service.parent_message("s1", "还有人在吗") is None


# ----------------------------------------------------------------------
# 5. 已转人工后仍回原固定文案(回归) + parent 落库
# ----------------------------------------------------------------------

class TestHandedOffStillFixedReply:

    def test_stream_fixed_reply_and_parent_message_logged(self, handoff_service, db_router):
        agent = _make_agent(handoff_service, session_id="s1")
        _collect(agent.classify_task_stream("转人工"))
        out = _collect(agent.classify_task_stream("我想约课"))
        assert out == "[REPLY]" + HANDED_OFF_REPLY
        # 家长消息落库
        msgs = db_router.handoff_repo.get_messages_by_session("s1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "parent"
        assert msgs[0]["content"] == "我想约课"

    def test_sync_fixed_reply_and_parent_message_logged(self, handoff_service, db_router):
        agent = _make_agent(handoff_service, session_id="s1")
        _run(agent.classify_task("转人工"))
        result = _run(agent.classify_task("你们在吗"))
        assert result == HANDED_OFF_REPLY
        msgs = db_router.handoff_repo.get_messages_by_session("s1")
        assert msgs and msgs[0]["role"] == "parent"


# ----------------------------------------------------------------------
# 6. 客服 API 端到端
# ----------------------------------------------------------------------

class TestHandoffLiveAPI:

    @pytest.fixture
    def identities(self, monkeypatch, tmp_path):
        """匿名身份服务指向临时文件库：家长侧 cookie 校验走 `api.identity` 单例，不能落到默认（生产）库。"""
        from api import identity as api_identity
        from services.identity_service import IdentityService

        service = IdentityService(f"sqlite:///{(tmp_path / 'handoff_live_identity.db').as_posix()}")
        monkeypatch.setattr(api_identity, "_identity_service", service)
        yield service
        service.close()

    @pytest.fixture
    def client(self, monkeypatch, tmp_path, identities):
        from fastapi.testclient import TestClient
        from app import app
        from services.human_handoff_service import HumanHandoffService

        db_file = tmp_path / "handoff_live_api.db"
        db_path = f"sqlite:///{db_file.as_posix()}"
        shared = {"router": None}

        def _patched_init(self, db_path_=None, threshold=None):
            from db.db_router import DatabaseRouter
            if shared["router"] is None:
                shared["router"] = DatabaseRouter(db_path)
            self.db_router = shared["router"]
            self.repo = self.db_router.handoff_repo
            self.threshold = self._resolve_threshold(threshold)
            self._kickback_counts = {}
            self._handed_off_sessions = set()

        monkeypatch.setattr(HumanHandoffService, "__init__", _patched_init)
        client = TestClient(app)
        _login_console(client, tmp_path, monkeypatch)
        yield client
        if shared["router"] is not None:
            shared["router"].close()

    def test_reply_then_fetch_messages(self, client, identities):
        from config.identity import IDENTITY_COOKIE_NAME
        from services.human_handoff_service import HumanHandoffService

        svc = HumanHandoffService()
        record = svc.trigger_handoff("api-live", "keyword", reason="人工")
        hid = record["id"]
        svc.parent_message("api-live", "大概什么时候能回复我")
        owner = identities.issue_new_identity()
        identities.create_session(owner, "api-live")

        r = client.post(f"/api/handoffs/{hid}/claim", json={"agent_id": "客服小美"})
        assert r.status_code == 200, r.text
        assert r.json()["data"]["agent_id"] == "客服小美"

        r2 = client.post(
            f"/api/handoffs/{hid}/reply",
            json={"agent_id": "客服小美", "content": "已为您预留周六试讲名额"},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["data"]["role"] == "agent"

        r3 = client.get(f"/api/handoffs/{hid}/messages")
        assert r3.status_code == 200, r3.text
        msgs = r3.json()["data"]
        assert [m["role"] for m in msgs] == ["parent", "agent"]
        assert msgs[0]["content"] == "大概什么时候能回复我"
        assert msgs[1]["content"] == "已为您预留周六试讲名额"
        # created_at 已转 ISO 字符串
        assert isinstance(msgs[0]["created_at"], str)

        client.cookies.set(IDENTITY_COOKIE_NAME, owner.user_id)
        r4 = client.get("/api/handoffs/session/api-live/messages")
        assert r4.status_code == 200, r4.text
        session_msgs = r4.json()["data"]
        assert [m["role"] for m in session_msgs] == ["parent", "agent"]

    def test_reply_unknown_returns_404(self, client):
        r = client.post(
            "/api/handoffs/999999/reply",
            json={"agent_id": "客服小美", "content": "hi"},
        )
        assert r.status_code == 404

    def test_claim_unknown_returns_404(self, client):
        r = client.post("/api/handoffs/999999/claim", json={"agent_id": "客服小美"})
        assert r.status_code == 404

    def test_reply_after_resolve_returns_404(self, client):
        from services.human_handoff_service import HumanHandoffService

        svc = HumanHandoffService()
        record = svc.trigger_handoff("api-resolved", "keyword", reason="人工")
        hid = record["id"]
        client.post(f"/api/handoffs/{hid}/resolve", json={"handled_by": "客服小美"})
        r = client.post(
            f"/api/handoffs/{hid}/reply",
            json={"agent_id": "客服小美", "content": "太晚了"},
        )
        assert r.status_code == 404

    def test_session_messages_endpoint_empty_for_unknown(self, client, identities):
        """未出现过的 session_id：带合法匿名身份 cookie → 200 + []（首次出现即绑定给当前身份，D 轮既有语义）。"""
        from config.identity import IDENTITY_COOKIE_NAME

        client.cookies.set(IDENTITY_COOKIE_NAME, identities.issue_new_identity().user_id)
        r = client.get("/api/handoffs/session/nobody/messages")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []


# ----------------------------------------------------------------------
# 7. 模型/导出存在性
# ----------------------------------------------------------------------

class TestSymbols:

    def test_human_handoff_message_exported(self):
        from db.models import HumanHandoffMessage
        from db import HumanHandoffMessage as Exported

        assert HumanHandoffMessage is Exported
        assert HumanHandoffMessage.__tablename__ == "human_handoff_messages"

    def test_service_has_live_methods(self):
        from services.human_handoff_service import HumanHandoffService

        for name in ("claim", "agent_reply", "parent_message", "snapshot_transcript",
                     "get_messages", "get_messages_by_session", "get_by_session"):
            assert hasattr(HumanHandoffService, name), f"缺少方法: {name}"
