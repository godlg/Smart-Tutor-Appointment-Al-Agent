"""
人工接管（Human Handoff）兜底测试

覆盖需求 P0:
- 关键词检测：主动要求转人工命中、无关输入不命中
- 主动触发流式：classify_task_stream("转人工") 输出含 [HANDOFF] + 人工文案,DB 落 keyword 记录
- 阈值触发：record_kickback 累计达阈值
- 阈值自动转人工：连续 3 轮 unrelated 输入 → 第 3 轮转人工,kickback_threshold 记录
- 已转人工短路：转人工后任意输入 → 短路文案,分类器零调用
- 幂等：同 session 重复触发只建 1 条 pending
- 阈值可配：HANDOFF_THRESHOLD=2 时 2 次即触发
- API：GET 列表返回 pending；POST resolve 后 resolved
- sync 路径：classify_task("转人工") 同样返回人工文案

约束：本仓库无 pytest-asyncio,异步用例一律用 _run(_collect) 同步包装;
不触网(手搓 fake 子 agent / fake classifier);不硬编码日期;DB 隔离(每用例独立 in-memory 库)。
"""
import asyncio
import os
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
        self.unsupported_calls = 0

    def get_available_services(self):
        return []

    async def handle_unsupported_task(self, category):
        self.unsupported_calls += 1
        yield "[REPLY][归类机器人]这个问题暂时超出我的能力范围。"


def _make_handoff_service(db_router, threshold=None):
    """在给定 db_router 上构造 HumanHandoffService(复用同一内存库)。"""
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
    """每用例独立 in-memory 数据库。"""
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


@pytest.fixture
def handoff_service(db_router):
    return _make_handoff_service(db_router)


def _make_agent(handoff_service, classifier=None, router=None):
    """手搓 TaskClassificationAgent:注入 fake classifier / router,不触网。

    直接改 processor 内部引用,绕过 TaskClassificationAgent.__init__ 的 create_chat_model。
    """
    from agents.task_classification_agent import TaskClassificationAgent

    agent = object.__new__(TaskClassificationAgent)
    agent.session_id = "test-session"
    agent.handoff_service = handoff_service

    processor = agent.classification_processor = _make_processor(
        handoff_service, classifier=classifier, router=router
    )
    # 让 agent 的短路逻辑与 processor 共用同一 classifier 计数
    agent._fake_classifier = processor.task_classifier
    agent._fake_router = processor.agent_router
    return agent


def _make_processor(handoff_service, classifier=None, router=None, session_id="test-session"):
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    classifier = classifier or _FakeClassifier("unrelated")
    router = router or _FakeRouter()
    state_manager = StateManager()
    processor = ClassificationProcessor(
        task_classifier=classifier,
        state_manager=state_manager,
        agent_router=router,
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=session_id,
    )
    return processor


# ----------------------------------------------------------------------
# 1. 关键词检测
# ----------------------------------------------------------------------

class TestKeywordMatch:

    def test_match_positive(self, handoff_service):
        hit = handoff_service.match_keyword("我要转人工")
        assert hit is not None
        assert hit in handoff_service.HANDOFF_KEYWORDS
        assert "转人工" in "我要转人工"

    def test_match_negative(self, handoff_service):
        assert handoff_service.match_keyword("今天天气不错") is None

    def test_keywords_cover_required(self, handoff_service):
        for kw in ("人工", "转人工", "真人", "人工客服", "人工服务",
                   "客服", "找经理", "投诉", "人工顾问", "人工咨询", "人在吗"):
            assert kw in handoff_service.HANDOFF_KEYWORDS, f"应覆盖关键词: {kw}"


# ----------------------------------------------------------------------
# 2. 主动触发（流式 + sync）
# ----------------------------------------------------------------------

class TestKeywordTrigger:

    def test_stream_keyword_triggers_handoff(self, handoff_service, db_router):
        agent = _make_agent(handoff_service)
        out = _collect(agent.classify_task_stream("转人工"))
        assert "[HANDOFF]" in out
        assert "已为您转接人工顾问老师" in out
        assert out.endswith(HANDOFF_REPLY)
        # 分类器零调用(keyword 短路先于分类)
        assert agent._fake_classifier.calls == 0
        # DB 落 keyword pending 记录
        pending = db_router.handoff_repo.get_pending()
        assert len(pending) == 1
        assert pending[0]["trigger_type"] == "keyword"
        assert pending[0]["status"] == "pending"

    def test_sync_keyword_returns_handoff_reply(self, handoff_service):
        agent = _make_agent(handoff_service)
        result = _run(agent.classify_task("转人工"))
        assert result == HANDOFF_REPLY
        assert agent._fake_classifier.calls == 0

    def test_stream_normal_input_does_not_trigger(self, handoff_service, db_router):
        agent = _make_agent(handoff_service, classifier=_FakeClassifier("consultation"))
        _collect(agent.classify_task_stream("我想约课"))
        assert agent._fake_classifier.calls == 1
        assert db_router.handoff_repo.get_pending() == []


# ----------------------------------------------------------------------
# 3. 踢回阈值计数
# ----------------------------------------------------------------------

class TestKickbackThreshold:

    def test_count_reaches_threshold_on_third(self, handoff_service):
        assert handoff_service.threshold == 3
        assert handoff_service.record_kickback("s1") is False
        assert handoff_service.record_kickback("s1") is False
        assert handoff_service.record_kickback("s1") is True

    def test_count_is_session_isolated(self, handoff_service):
        handoff_service.record_kickback("s1")
        handoff_service.record_kickback("s1")
        # 另一个会话独立计数,不受影响
        assert handoff_service.record_kickback("s2") is False

    def test_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("HANDOFF_THRESHOLD", "2")
        from services.human_handoff_service import HumanHandoffService
        svc = HumanHandoffService.__new__(HumanHandoffService)
        assert svc._resolve_threshold(None) == 2

    def test_count_configured_via_constructor(self, monkeypatch):
        """显式传 threshold 优先于 env。"""
        monkeypatch.setenv("HANDOFF_THRESHOLD", "5")
        from services.human_handoff_service import HumanHandoffService
        assert HumanHandoffService._resolve_threshold(2) == 2


# ----------------------------------------------------------------------
# 4. 阈值自动转人工
# ----------------------------------------------------------------------

class TestAutoHandoffByThreshold:

    def test_three_rounds_trigger(self, handoff_service, db_router):
        agent = _make_agent(handoff_service, classifier=_FakeClassifier("unrelated"))
        r1 = _collect(agent.classify_task_stream("今天天气怎么样"))
        r2 = _collect(agent.classify_task_stream("随便聊聊"))
        # 前两轮仍是能力列表,不转人工
        assert "[HANDOFF]" not in r1
        assert "[HANDOFF]" not in r2
        r3 = _collect(agent.classify_task_stream("再问个无关的"))
        assert "[HANDOFF]" in r3
        assert r3.endswith(HANDOFF_REPLY)

        pending = db_router.handoff_repo.get_pending()
        assert len(pending) == 1
        assert pending[0]["trigger_type"] == "kickback_threshold"
        assert pending[0]["unrelated_count"] >= 3

    def test_sync_three_rounds_trigger(self, handoff_service, db_router):
        agent = _make_agent(handoff_service, classifier=_FakeClassifier("unrelated"))
        _run(agent.classify_task("无关一"))
        _run(agent.classify_task("无关二"))
        r3 = _run(agent.classify_task("无关三"))
        assert "[HANDOFF]" in r3
        assert db_router.handoff_repo.get_pending()[0]["trigger_type"] == "kickback_threshold"

    def test_threshold_configurable_two_triggers(self, db_router):
        svc = _make_handoff_service(db_router, threshold=2)
        agent = _make_agent(svc, classifier=_FakeClassifier("unrelated"))
        r1 = _collect(agent.classify_task_stream("无关一"))
        assert "[HANDOFF]" not in r1
        r2 = _collect(agent.classify_task_stream("无关二"))
        assert "[HANDOFF]" in r2


# ----------------------------------------------------------------------
# 5. 已转人工短路（分类器零调用）
# ----------------------------------------------------------------------

class TestHandedOffShortCircuit:

    def test_after_handoff_any_input_short_circuits(self, handoff_service):
        agent = _make_agent(handoff_service)
        _collect(agent.classify_task_stream("转人工"))
        calls_after_handoff = agent._fake_classifier.calls

        out = _collect(agent.classify_task_stream("我想约课"))
        assert out == "[REPLY]" + HANDED_OFF_REPLY
        # 短路:分类器零新增调用
        assert agent._fake_classifier.calls == calls_after_handoff == 0

    def test_sync_after_handoff_short_circuits(self, handoff_service):
        agent = _make_agent(handoff_service)
        _run(agent.classify_task("转人工"))
        result = _run(agent.classify_task("我想约课"))
        assert result == HANDED_OFF_REPLY

    def test_is_handed_off_falls_back_to_db(self, db_router):
        """另一实例(内存 set 为空)也能通过 DB pending 记录识别已转人工。"""
        svc_a = _make_handoff_service(db_router)
        svc_a.trigger_handoff("sess-x", "keyword", reason="人工")

        svc_b = _make_handoff_service(db_router)
        assert svc_b.is_handed_off("sess-x") is True


# ----------------------------------------------------------------------
# 6. 幂等：同 session 重复触发只建 1 条 pending
# ----------------------------------------------------------------------

class TestIdempotent:

    def test_repeat_trigger_single_pending(self, handoff_service, db_router):
        first = handoff_service.trigger_handoff("sess-1", "keyword", reason="人工")
        second = handoff_service.trigger_handoff("sess-1", "keyword", reason="转人工")
        assert first["id"] == second["id"]
        assert len(db_router.handoff_repo.get_pending()) == 1

    def test_second_handoff_call_does_not_duplicate(self, handoff_service, db_router):
        agent = _make_agent(handoff_service)
        _collect(agent.classify_task_stream("转人工"))
        _collect(agent.classify_task_stream("转人工"))
        pending = db_router.handoff_repo.get_pending()
        assert len(pending) == 1


# ----------------------------------------------------------------------
# 7. 客服 API
# ----------------------------------------------------------------------

class TestHandoffAPI:

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        """TestClient + 把 HumanHandoffService 强制指向临时文件库。

        用文件库(而非 :memory:)是因为 TestClient 在独立线程跑 app,
        :memory: 每个连接都是独立库,跨线程查不到表。
        """
        from fastapi.testclient import TestClient
        from app import app
        from services.human_handoff_service import HumanHandoffService

        db_file = tmp_path / "handoff_api.db"
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
        yield client
        if shared["router"] is not None:
            shared["router"].close()

    def test_get_pending_then_resolve(self, client):
        from services.human_handoff_service import HumanHandoffService

        svc = HumanHandoffService()
        svc.trigger_handoff("api-sess", "keyword", reason="人工")

        r = client.get("/api/handoffs?status=pending")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"]
        handoff_id = body["data"][0]["id"]
        assert body["data"][0]["status"] == "pending"

        r2 = client.post(f"/api/handoffs/{handoff_id}/resolve",
                         json={"handled_by": "客服A", "note": "已回电"})
        assert r2.status_code == 200, r2.text
        assert r2.json()["data"]["status"] == "resolved"
        assert r2.json()["data"]["handled_by"] == "客服A"

        # resolve 后不再出现在 pending 列表
        r3 = client.get("/api/handoffs?status=pending")
        assert all(item["id"] != handoff_id for item in r3.json()["data"])

    def test_resolve_unknown_returns_404(self, client):
        r = client.post("/api/handoffs/999999/resolve", json={})
        assert r.status_code == 404


# ----------------------------------------------------------------------
# 8. 固定文案存在性
# ----------------------------------------------------------------------

class TestReplyTexts:

    def test_reply_texts_fixed(self, handoff_service):
        assert handoff_service.handoff_reply() == HANDOFF_REPLY
        assert handoff_service.handed_off_reply() == HANDED_OFF_REPLY


# ----------------------------------------------------------------------
# 9. 阈值计数动态日期无关（防御性:确保测试不硬编码日期）
# ----------------------------------------------------------------------

class TestNoHardcodedDate:

    def test_uses_dynamic_time(self):
        from datetime import datetime
        # 触发记录 created_at 应为当前时间(动态),而非固定日期
        from db.db_router import DatabaseRouter
        router = DatabaseRouter("sqlite:///:memory:")
        try:
            svc = _make_handoff_service(router)
            before = datetime.utcnow()
            svc.trigger_handoff("date-sess", "keyword", reason="人工")
            record = router.handoff_repo.get_pending()[0]
            assert record["created_at"] >= before
        finally:
            router.close()
