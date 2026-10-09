"""
LLM 分类故障降级测试。

覆盖修复:LLM 分类服务不可用(超时 / 限流 429 / 网络 / 接口错误)时,
旧实现静默返回 "unrelated" → 落入兜底 → record_kickback 累计达阈值自动转人工,
家长连正常业务输入都被判无关并在第 3 轮被强制转人工(实测见任务文档)。

本套件验证:
- TaskClassifier.classify_task 在 LLM 异常时抛 ClassificationUnavailable(不再返回 unrelated);
  ContextBudgetExceeded 仍原样透传。
- 流式 / 同步路径捕获 ClassificationUnavailable → 回「系统繁忙」,不踢回、不转人工、不改状态。
- 对照组:LLM 正常返回 unrelated 时,原踢回计数 → 阈值转人工逻辑保持不变。

约束:本仓库无 pytest-asyncio,异步用例一律用 _run 同步包装;
不触网(手搓 fake 子 agent / fake classifier);DB 隔离(每用例独立 in-memory 库)。
"""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from agents.context.types import ContextBudgetExceeded
from agents.task_classification.task_classifier import (
    TaskClassifier,
    ClassificationUnavailable,
)
from agents.task_classification.classification_processor import (
    ClassificationProcessor,
    _BUSY_REPLY,
)
from agents.task_classification.state_manager import StateManager, FlowState

HANDOFF_MARKER = "[HANDOFF]"


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    out = []
    async for token in agen:
        out.append(token)
    return "".join(out)


def _collect(agen):
    return _run(_drain(agen))


class _AccountQuotaExceeded(Exception):
    """模拟 LLM 服务端 429 型异常(实测复现中的真实类名)。"""


class _RaisingClassifier:
    """打桩分类器:classify_task 固定抛 ClassificationUnavailable,并记录调用次数。"""

    def __init__(self):
        self.calls = 0

    async def classify_task(self, task):
        self.calls += 1
        raise ClassificationUnavailable("AccountQuotaExceeded: 429")


class _FixedClassifier:
    """打桩分类器:固定返回指定分类结果,并记录调用次数。"""

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


@pytest.fixture
def db_router():
    """每用例独立 in-memory 数据库。"""
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


def _make_handoff_service(db_router, threshold=None):
    """在给定 db_router 上构造 HumanHandoffService(复用同一内存库),并给关键方法挂调用探针。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService.__new__(HumanHandoffService)
    svc.db_router = db_router
    svc.repo = db_router.handoff_repo
    svc.threshold = svc._resolve_threshold(threshold)
    svc._kickback_counts = {}
    svc._handed_off_sessions = set()
    svc._recent_inputs = {}
    svc.trigger_handoff_spy = Mock(wraps=svc.trigger_handoff)
    svc.record_kickback_spy = Mock(wraps=svc.record_kickback)
    svc.trigger_handoff = svc.trigger_handoff_spy
    svc.record_kickback = svc.record_kickback_spy
    return svc


def _make_processor(handoff_service, classifier, router=None, session_id="test-session"):
    """手搓 ClassificationProcessor:注入打桩 classifier / router,不触网。"""
    return ClassificationProcessor(
        task_classifier=classifier,
        state_manager=StateManager(),
        agent_router=router or _FakeRouter(),
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=session_id,
    )


# ----------------------------------------------------------------------
# 1. TaskClassifier:LLM 故障 → ClassificationUnavailable
# ----------------------------------------------------------------------

def _failing_classifier(exc):
    """构造分类器,令其链在 ainvoke 时抛 exc。"""
    clf = TaskClassifier.__new__(TaskClassifier)
    clf.context_manager = None  # 未注入时走旧 PromptTemplate 链
    clf.chain = Mock()
    clf.chain.ainvoke = AsyncMock(side_effect=exc)
    return clf


class TestClassifierRaisesUnavailable:

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("模型服务异常"),
            TimeoutError("请求超时"),
            _AccountQuotaExceeded("AccountQuotaExceeded: 429"),
        ],
    )
    def test_llm_failure_raises_classification_unavailable(self, exc):
        clf = _failing_classifier(exc)
        with pytest.raises(ClassificationUnavailable):
            _run(clf.classify_task("我想给初三的孩子约一节数学试听课"))

    def test_no_longer_returns_unrelated_on_failure(self):
        """回归保护:故障绝不能再被伪装成 unrelated。"""
        clf = _failing_classifier(RuntimeError("boom"))
        try:
            result = _run(clf.classify_task("我想约课"))
        except ClassificationUnavailable:
            return
        pytest.fail(f"故障时不应返回分类结果,实际返回: {result!r}")

    def test_context_budget_exceeded_passes_through(self):
        """ContextBudgetExceeded 必须原样透传,不能被包装成 ClassificationUnavailable。"""
        clf = _failing_classifier(ContextBudgetExceeded("超预算"))
        with pytest.raises(ContextBudgetExceeded):
            _run(clf.classify_task("超长输入"))

    def test_original_exception_chained(self):
        """异常链保留原异常,便于日志定位。"""
        original = _AccountQuotaExceeded("AccountQuotaExceeded: 429")
        clf = _failing_classifier(original)
        with pytest.raises(ClassificationUnavailable) as excinfo:
            _run(clf.classify_task("我想约课"))
        assert excinfo.value.__cause__ is original


# ----------------------------------------------------------------------
# 2. 流式路径:连续 5 次故障 → 只回繁忙,不踢回 / 不转人工 / 不改状态
# ----------------------------------------------------------------------

class TestStreamFailureFallback:

    def test_five_failures_never_handoff(self, db_router):
        handoff = _make_handoff_service(db_router)
        classifier = _RaisingClassifier()
        processor = _make_processor(handoff, classifier)

        for i in range(5):
            out = _collect(processor.process_task_stream("我想给初三的孩子约一节数学试听课"))
            assert _BUSY_REPLY in out, f"第 {i + 1} 次应回繁忙文案"
            assert HANDOFF_MARKER not in out, f"第 {i + 1} 次不应转人工"

        assert classifier.calls == 5
        handoff.trigger_handoff_spy.assert_not_called()
        handoff.record_kickback_spy.assert_not_called()
        assert handoff._kickback_counts.get("test-session", 0) == 0
        assert processor.state_manager.get_current_state() == FlowState.CLASSIFY

    def test_five_failures_leave_no_pending_handoff(self, db_router):
        handoff = _make_handoff_service(db_router)
        processor = _make_processor(handoff, _RaisingClassifier())
        for _ in range(5):
            _collect(processor.process_task_stream("我想约课"))
        assert db_router.handoff_repo.get_pending() == []

    def test_busy_reply_has_reply_prefix(self, db_router):
        handoff = _make_handoff_service(db_router)
        processor = _make_processor(handoff, _RaisingClassifier())
        out = _collect(processor.process_task_stream("我想约课"))
        assert out.startswith("[REPLY]")


# ----------------------------------------------------------------------
# 3. 同步路径:连续 5 次故障 → 只回繁忙,不踢回 / 不转人工 / 不改状态
# ----------------------------------------------------------------------

class TestSyncFailureFallback:

    def test_five_failures_never_handoff(self, db_router):
        handoff = _make_handoff_service(db_router)
        classifier = _RaisingClassifier()
        processor = _make_processor(handoff, classifier)

        for i in range(5):
            out = _run(processor.process_task_sync("我想给初三的孩子约一节数学试听课"))
            assert out == _BUSY_REPLY, f"第 {i + 1} 次应回繁忙文案"
            assert HANDOFF_MARKER not in out

        assert classifier.calls == 5
        handoff.trigger_handoff_spy.assert_not_called()
        handoff.record_kickback_spy.assert_not_called()
        assert handoff._kickback_counts.get("test-session", 0) == 0
        assert processor.state_manager.get_current_state() == FlowState.CLASSIFY
        assert db_router.handoff_repo.get_pending() == []


# ----------------------------------------------------------------------
# 4. 对照组:真 unrelated 仍走踢回 → 阈值转人工(原行为不被破坏)
# ----------------------------------------------------------------------

class TestRealUnrelatedStillHandoffs:

    def test_three_real_unrelated_rounds_trigger_handoff_stream(self, db_router):
        handoff = _make_handoff_service(db_router)
        classifier = _FixedClassifier("unrelated")
        processor = _make_processor(handoff, classifier)

        r1 = _collect(processor.process_task_stream("今天天气怎么样"))
        r2 = _collect(processor.process_task_stream("随便聊聊"))
        r3 = _collect(processor.process_task_stream("再问个无关的"))

        assert HANDOFF_MARKER not in r1
        assert HANDOFF_MARKER not in r2
        assert HANDOFF_MARKER in r3
        assert handoff.record_kickback_spy.call_count == 3
        assert handoff.trigger_handoff_spy.call_count == 1
        pending = db_router.handoff_repo.get_pending()
        assert len(pending) == 1
        assert pending[0]["trigger_type"] == "kickback_threshold"

    def test_three_real_unrelated_rounds_trigger_handoff_sync(self, db_router):
        handoff = _make_handoff_service(db_router)
        classifier = _FixedClassifier("unrelated")
        processor = _make_processor(handoff, classifier)

        _run(processor.process_task_sync("无关一"))
        _run(processor.process_task_sync("无关二"))
        r3 = _run(processor.process_task_sync("无关三"))

        assert HANDOFF_MARKER in r3
        assert db_router.handoff_repo.get_pending()[0]["trigger_type"] == "kickback_threshold"

    def test_unrelated_does_not_use_busy_reply(self, db_router):
        """真无关不能被伪装成系统繁忙。"""
        handoff = _make_handoff_service(db_router)
        processor = _make_processor(handoff, _FixedClassifier("unrelated"))
        out = _collect(processor.process_task_stream("今天天气怎么样"))
        assert _BUSY_REPLY not in out
