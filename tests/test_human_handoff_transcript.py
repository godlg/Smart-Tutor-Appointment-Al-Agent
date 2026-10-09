"""
人工接管 P1.1：transcript 补全（无关轮输入也进快照）测试

背景:无关轮(判 unrelated → 只 yield 能力列表)不写会话历史,转人工时
snapshot_transcript 快照不到家长原话。新增进程内"最近用户输入"缓冲
(record_input) + build_transcript(历史快照 + 缓冲补全)修复。

覆盖需求:
- 无关轮进缓冲:连续 3 轮 unrelated,第 3 轮触发后 transcript 含全部 3 条用户输入
- keyword 触发同样含最近输入
- 去重:history 已有某条内容 + 缓冲含相同 content → transcript 不重复
- 截断:连发 7 条 → 缓冲只留最近 5 条(触发后 transcript 只含 5 条)
- 混合:history 2 条业务 + 缓冲 2 条无关 → 4 条(history 在前)
- 符号存在:record_input / build_transcript / RECENT_INPUTS_MAXLEN

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


# ----------------------------------------------------------------------
# 测试替身(不触网)
# ----------------------------------------------------------------------

class _FakeClassifier:
    """固定返回 unrelated,记录调用次数(不依赖真实 LLM)。"""

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
    svc._recent_inputs = {}
    return svc


def _make_agent(handoff_service, classifier=None, router=None,
                session_id="test-session", session_chat_history=None):
    """手搓 TaskClassificationAgent:注入 fake,不触网。"""
    from agents.task_classification_agent import TaskClassificationAgent
    from agents.task_classification.classification_processor import ClassificationProcessor
    from agents.task_classification.state_manager import StateManager

    agent = object.__new__(TaskClassificationAgent)
    agent.session_id = session_id
    agent.handoff_service = handoff_service
    agent.session_chat_history = session_chat_history
    processor = agent.classification_processor = ClassificationProcessor(
        task_classifier=classifier or _FakeClassifier("unrelated"),
        state_manager=StateManager(),
        agent_router=router or _FakeRouter(),
        unrelated_handler=None,
        handoff_service=handoff_service,
        session_id=session_id,
        session_chat_history=session_chat_history,
    )
    agent._fake_classifier = processor.task_classifier
    agent._fake_router = processor.agent_router
    return agent


@pytest.fixture
def db_router():
    from db.db_router import DatabaseRouter

    router = DatabaseRouter("sqlite:///:memory:")
    yield router
    router.close()


@pytest.fixture
def handoff_service(db_router):
    return _make_handoff_service(db_router)


def _contents(transcript):
    return [item["content"] for item in json.loads(transcript)]


# ----------------------------------------------------------------------
# 1. 无关轮进缓冲:3 轮 unrelated,第 3 轮触发含全部 3 条用户输入
# ----------------------------------------------------------------------

class TestUnrelatedRoundsInTranscript:

    def test_three_unrelated_rounds_all_in_transcript(self, handoff_service, db_router):
        history = _make_history([
            _FakeMessage("human", "我想给高三孩子找数学老师"),
            _FakeMessage("ai", "好的,方便说下预算吗"),
        ])
        agent = _make_agent(handoff_service, session_chat_history=history)

        inputs = ["今天天气怎么样", "给我讲个笑话", "随便聊聊"]
        _collect(agent.classify_task_stream(inputs[0]))
        _collect(agent.classify_task_stream(inputs[1]))
        out = _collect(agent.classify_task_stream(inputs[2]))

        assert "[HANDOFF]" in out, "第 3 轮无关输入应触发转人工"
        record = db_router.handoff_repo.get_pending()[0]
        assert record["trigger_type"] == "kickback_threshold"
        parsed = json.loads(record["transcript"])
        contents = [item["content"] for item in parsed]
        # 3 条无关轮原话都在快照里(无关轮不写 history,靠缓冲补全)
        for text in inputs:
            assert text in contents, f"缺少无关轮输入: {text}"
        assert all(
            item["role"] == "human"
            for item in parsed if item["content"] in inputs
        )

    def test_sync_three_unrelated_rounds_contain_inputs(self, handoff_service, db_router):
        agent = _make_agent(handoff_service)
        _run(agent.classify_task("无关甲"))
        _run(agent.classify_task("无关乙"))
        _run(agent.classify_task("无关丙"))
        record = db_router.handoff_repo.get_pending()[0]
        assert _contents(record["transcript"]) == ["无关甲", "无关乙", "无关丙"]


# ----------------------------------------------------------------------
# 2. keyword 触发含最近无关输入
# ----------------------------------------------------------------------

class TestKeywordTriggerIncludesRecentInputs:

    def test_keyword_transcript_keeps_prior_unrelated_inputs(self, handoff_service, db_router):
        history = _make_history([_FakeMessage("ai", "您好,有什么可以帮您")])
        agent = _make_agent(handoff_service, session_chat_history=history)

        _collect(agent.classify_task_stream("今天天气怎么样"))
        _collect(agent.classify_task_stream("推荐部电影"))
        out = _collect(agent.classify_task_stream("转人工"))

        assert "[HANDOFF]" in out
        record = db_router.handoff_repo.get_pending()[0]
        assert record["trigger_type"] == "keyword"
        contents = _contents(record["transcript"])
        assert "今天天气怎么样" in contents
        assert "推荐部电影" in contents
        assert "转人工" in contents


# ----------------------------------------------------------------------
# 3. 去重:history 与缓冲同内容不重复
# ----------------------------------------------------------------------

class TestDedup:

    def test_same_content_in_history_and_buffer_appears_once(self, handoff_service):
        history = _make_history([
            _FakeMessage("human", "我想约课"),
            _FakeMessage("ai", "好的"),
        ])
        handoff_service.record_input("s1", "我想约课")
        handoff_service.record_input("s1", "另外问下停车")
        parsed = json.loads(handoff_service.build_transcript("s1", history.messages))
        contents = [item["content"] for item in parsed]
        assert contents.count("我想约课") == 1
        assert contents == ["我想约课", "好的", "另外问下停车"]


# ----------------------------------------------------------------------
# 4. 截断:连发 7 条只留最近 5 条
# ----------------------------------------------------------------------

class TestBufferTruncation:

    def test_seven_inputs_transcript_keeps_last_five(self, db_router):
        # 阈值设 7:第 7 轮才触发,触发时缓冲刚好被 maxlen=5 截断
        svc = _make_handoff_service(db_router, threshold=7)
        agent = _make_agent(svc, session_chat_history=None)
        inputs = [f"无关{ i }" for i in range(1, 8)]
        for i, text in enumerate(inputs, start=1):
            out = _collect(agent.classify_task_stream(text))
        assert "[HANDOFF]" in out, "第 7 轮应达阈值触发"
        record = db_router.handoff_repo.get_pending()[0]
        contents = _contents(record["transcript"])
        assert contents == inputs[-5:], "缓冲只应保留最近 5 条"
        assert len(contents) == 5

    def test_record_input_directly_truncates(self, handoff_service):
        from services.human_handoff_service import RECENT_INPUTS_MAXLEN

        for i in range(1, 8):
            handoff_service.record_input("s1", f"输入{i}")
        parsed = json.loads(handoff_service.build_transcript("s1", None))
        assert len(parsed) == RECENT_INPUTS_MAXLEN == 5
        assert [item["content"] for item in parsed] == ["输入3", "输入4", "输入5", "输入6", "输入7"]


# ----------------------------------------------------------------------
# 5. 混合:history 2 条业务 + 缓冲 2 条无关 = 4 条(history 在前)
# ----------------------------------------------------------------------

class TestMixedHistoryAndBuffer:

    def test_history_first_then_buffer(self, handoff_service):
        history = _make_history([
            _FakeMessage("human", "孩子初三数学"),
            _FakeMessage("ai", "好的,已记录"),
        ])
        handoff_service.record_input("s1", "今天天气怎么样")
        handoff_service.record_input("s1", "给我讲个笑话")
        parsed = json.loads(handoff_service.build_transcript("s1", history.messages))
        assert [item["content"] for item in parsed] == [
            "孩子初三数学", "好的,已记录", "今天天气怎么样", "给我讲个笑话",
        ]
        assert parsed[0]["role"] == "human"
        assert parsed[1]["role"] == "ai"
        assert parsed[2]["role"] == "human"
        assert parsed[3]["role"] == "human"


# ----------------------------------------------------------------------
# 6. 边界
# ----------------------------------------------------------------------

class TestBuildTranscriptEdges:

    def test_empty_history_and_buffer_returns_none(self, handoff_service):
        assert handoff_service.build_transcript("nobody", None) is None
        assert handoff_service.build_transcript("nobody", []) is None

    def test_none_content_ignored(self, handoff_service):
        handoff_service.record_input("s1", None)
        assert handoff_service.build_transcript("s1", None) is None

    def test_sessions_isolated(self, handoff_service):
        handoff_service.record_input("s-a", "A 的问题")
        handoff_service.record_input("s-b", "B 的问题")
        assert _contents(handoff_service.build_transcript("s-a", None)) == ["A 的问题"]
        assert _contents(handoff_service.build_transcript("s-b", None)) == ["B 的问题"]

    def test_new_instance_without_buffer_attr_still_works(self, handoff_service):
        # 模拟 __new__ 手搓实例未初始化 _recent_inputs
        del handoff_service._recent_inputs
        handoff_service.record_input("s1", "你好")
        assert _contents(handoff_service.build_transcript("s1", None)) == ["你好"]


# ----------------------------------------------------------------------
# 7. 关键符号存在性
# ----------------------------------------------------------------------

class TestSymbols:

    def test_service_has_buffer_symbols(self):
        from services.human_handoff_service import HumanHandoffService, RECENT_INPUTS_MAXLEN

        assert RECENT_INPUTS_MAXLEN == 5
        assert hasattr(HumanHandoffService, "record_input")
        assert hasattr(HumanHandoffService, "build_transcript")

    def test_new_instance_initializes_buffer(self, tmp_path):
        from services.human_handoff_service import HumanHandoffService

        db_file = tmp_path / "buf_init.db"
        svc = HumanHandoffService(f"sqlite:///{db_file.as_posix()}")
        assert isinstance(svc._recent_inputs, dict)
        svc.db_router.close()
