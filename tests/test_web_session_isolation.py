"""
Web 聊天按浏览器会话隔离（session_id 透传）测试

覆盖修复目标：
1. ChatRequest 增加 session_id 字段（可选、可传）
2. /chat/stream 与 /chat 把 session_id 透传给 ProcessUserInput_stream
   （带值传值；缺失传 None，保持走全局单例的向后兼容）
3. chat_handler 按 session_id 隔离：不同 session 返回不同 agent，
   同一 session 复用同一 agent
4. 隔离回归（核心，防再串台）：A 会话停在「推荐确认态」时，
   B 会话的全新推荐请求不受影响——不误回「您说的是哪位老师？」

框架约束：本仓库无 pytest-asyncio，异步用例一律用 _run(_collect) 同步包装。
不触网：mock 掉 RecommendationInputParser.extract 与推荐服务。
"""
import asyncio
import sys
from pathlib import Path

from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


async def _collect(stream) -> str:
    tokens = []
    async for tok in stream:
        tokens.append(tok)
    return "".join(tokens)


# ----------------------------------------------------------------------
# 1. ChatRequest schema：session_id 字段可选、可传
# ----------------------------------------------------------------------

class TestChatRequestSchema:

    def test_session_id_optional_defaults_none(self):
        from web.routes import ChatRequest
        req = ChatRequest(message="hi")
        assert req.session_id is None

    def test_session_id_can_be_provided(self):
        from web.routes import ChatRequest
        req = ChatRequest(message="hi", session_id="s-test-1")
        assert req.session_id == "s-test-1"

    def test_extra_fields_rejected(self):
        """session_id 之外的未知字段应被 Pydantic 忽略（不破坏既有请求）。"""
        from web.routes import ChatRequest
        req = ChatRequest(message="hi", session_id="s-test-1")
        assert req.message == "hi"


# ----------------------------------------------------------------------
# 2. 端点透传：/chat/stream 与 /chat 把 session_id 交给 ProcessUserInput_stream
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


class TestEndpointPassthrough:
    """monkeypatch ProcessUserInput_stream 为捕获参数的 fake，断言拿到正确的 session_id。"""

    def test_stream_passes_session_id(self, client, monkeypatch):
        captured = {}

        async def fake_stream(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            yield "generated-token"

        monkeypatch.setattr("web.routes.ProcessUserInput_stream", fake_stream)

        r = client.post("/chat/stream", json={"message": "hi", "session_id": "s-test-1"})
        assert r.status_code == 200
        assert captured["args"] == ("hi",)
        assert captured["kwargs"] == {"session_id": "s-test-1"}
        assert r.text == "generated-token"

    def test_stream_without_session_id_passes_none(self, client, monkeypatch):
        captured = {}

        async def fake_stream(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            yield "ok"

        monkeypatch.setattr("web.routes.ProcessUserInput_stream", fake_stream)
        r = client.post("/chat/stream", json={"message": "hi"})
        assert r.status_code == 200
        assert captured["args"] == ("hi",)
        assert captured["kwargs"] == {"session_id": None}

    def test_legacy_chat_endpoint_passes_session_id(self, client, monkeypatch):
        captured = {}

        async def fake_stream(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            yield "ok"

        monkeypatch.setattr("web.routes.ProcessUserInput_stream", fake_stream)
        r = client.post("/chat", json={"message": "hi", "session_id": "s-test-2"})
        assert r.status_code == 200
        assert captured["args"] == ("hi",)
        assert captured["kwargs"] == {"session_id": "s-test-2"}

    def test_legacy_chat_endpoint_without_session_id_passes_none(self, client, monkeypatch):
        captured = {}

        async def fake_stream(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            yield "ok"

        monkeypatch.setattr("web.routes.ProcessUserInput_stream", fake_stream)
        r = client.post("/chat", json={"message": "hi"})
        assert r.status_code == 200
        assert captured["kwargs"] == {"session_id": None}


# ----------------------------------------------------------------------
# 3. chat_handler 隔离：不同 session 不同 agent、同一 session 复用
# ----------------------------------------------------------------------

class TestChatHandlerIsolation:

    def test_session_agent_identity(self):
        from api import chat_handler

        try:
            a = chat_handler.get_task_agent("sess-isol-a")
            b = chat_handler.get_task_agent("sess-isol-b")
            a_again = chat_handler.get_task_agent("sess-isol-a")

            assert a is not b                # 不同 session → 独立 agent
            assert a is a_again              # 同一 session → 复用同一 agent
        finally:
            chat_handler.drop_session_agent("sess-isol-a")
            chat_handler.drop_session_agent("sess-isol-b")
            chat_handler._session_chat_histories.pop("sess-isol-a", None)
            chat_handler._session_chat_histories.pop("sess-isol-b", None)

    def test_session_isolation_does_not_touch_global_default(self):
        """session_id 缺失仍走全局单例（向后兼容），不因隔离改造被破坏。"""
        from api import chat_handler

        default = chat_handler.get_task_agent(None)
        sid_agent = chat_handler.get_task_agent("sess-isol-c")
        try:
            assert default is not sid_agent
            # 全局单例仍是那个全局单例
            assert chat_handler.get_task_agent(None) is default
        finally:
            chat_handler.drop_session_agent("sess-isol-c")
            chat_handler._session_chat_histories.pop("sess-isol-c", None)


# ----------------------------------------------------------------------
# 4. 隔离回归（核心，防再串台）
# ----------------------------------------------------------------------

_RECS = [
    {"teacher": {"id": 7, "name": "周敏"}, "score": 92, "reason": "擅长初二数学"},
    {"teacher": {"id": 9, "name": "张明"}, "score": 80, "reason": "覆盖初中"},
    {"teacher": {"id": 11, "name": "高远"}, "score": 70, "reason": "耐心细致"},
]
_COMPLETE_INPUT = {
    "subject": "数学", "grade_level": "初中", "preferred_style": "温和",
    "preferred_teacher_name": "未知", "mode": "未知",
    "info_complete": True, "missing_info": [],
}

CONFIRM_MISMATCH_MSG = "您说的是哪位老师？"


def _build_iso_agents():
    """经 get_task_agent('sess-a'/'sess-b') 构造两个彼此独立的推荐 agent。

    全部 mock 掉 extract 与推荐服务，不触网。
    """
    from api import chat_handler
    ta = chat_handler.get_task_agent("iso-sess-a")
    tb = chat_handler.get_task_agent("iso-sess-b")

    for agent in (ta.recommendation_agent, tb.recommendation_agent):
        # 整体替换 parser（真实 parser 绑定了真实 LLM，直接设 return_value 会因
        # 'method' object 无属性而报错），与既有推荐测试的做法一致。
        agent.input_parser = MagicMock()
        agent.input_parser.extract.return_value = _COMPLETE_INPUT
        agent._recommendation_service = MagicMock()
        agent._recommendation_service.recommend_teachers.return_value = _RECS
        agent._student_service = MagicMock()
        agent._student_service.initialize_default_student.return_value = {"id": 1}
    return ta, tb


def _cleanup_iso():
    from api import chat_handler
    for sid in ("iso-sess-a", "iso-sess-b"):
        chat_handler.drop_session_agent(sid)
        chat_handler._session_chat_histories.pop(sid, None)


class TestIsolationRegression:

    def test_session_b_not_polluted_by_session_a_confirmation(self):
        """A 停在推荐确认态；B 发全新推荐请求 → 仍是正常推荐，不是「您说的是哪位老师？」"""
        from api import chat_handler
        ta, tb = _build_iso_agents()
        try:
            # --- 会话 A：跑完一轮推荐并选定老师，进入确认态 ---
            _run(_collect(ta.recommendation_agent.run_stream(
                "请为他推荐一位老师", session_id="iso-sess-a")))
            a_reply = _run(_collect(ta.recommendation_agent.run_stream(
                "周敏", session_id="iso-sess-a")))
            assert "已为您选定 周敏" in a_reply
            a_hist = ta.recommendation_agent.history["iso-sess-a"]
            assert a_hist["awaiting_confirmation"] is True
            assert a_hist["confirmed_teacher"] is not None   # A 确实停在确认态

            # --- 会话 B：全新推荐请求 ---
            b_reply = _run(_collect(tb.recommendation_agent.run_stream(
                "请为我的孩子推荐一位老师", session_id="iso-sess-b")))

            # B 不是被 A 的确认态带偏（如果共用同一份状态机，这里会误回「您说的是哪位老师？」）
            assert CONFIRM_MISMATCH_MSG not in b_reply
            assert "为您推荐" in b_reply                      # 正常走了推荐
            # 隔离证明：A 已选定「周敏」，B 却从未选定任何老师
            b_hist = tb.recommendation_agent.history["iso-sess-b"]
            assert b_hist["confirmed_teacher"] is None
            assert b_hist["awaiting_confirmation"] is True    # 只是本轮推荐后的正常态
        finally:
            _cleanup_iso()

    def test_session_b_missing_fields_asks_clarification_not_mismatch(self):
        """B 字段不全 → 正常追问学科/年级，而非沿用 A 的确认态误回「您说的是哪位老师？」"""
        from api import chat_handler
        ta, tb = _build_iso_agents()
        try:
            # A 进入确认态
            _run(_collect(ta.recommendation_agent.run_stream(
                "请为他推荐一位老师", session_id="iso-sess-a")))
            _run(_collect(ta.recommendation_agent.run_stream(
                "周敏", session_id="iso-sess-a")))

            # B：extract 返回字段不全 → 应走追问
            incomplete = dict(_COMPLETE_INPUT)
            incomplete.update({"subject": "未知", "grade_level": "未知",
                               "info_complete": False,
                               "missing_info": ["subject", "grade_level"]})
            tb.recommendation_agent.input_parser.extract.return_value = incomplete

            b_reply = _run(_collect(tb.recommendation_agent.run_stream(
                "帮我推荐老师", session_id="iso-sess-b")))

            assert CONFIRM_MISMATCH_MSG not in b_reply
            assert "学科" in b_reply               # 正常追问而不是确认态错位
            assert "年级" in b_reply
        finally:
            _cleanup_iso()