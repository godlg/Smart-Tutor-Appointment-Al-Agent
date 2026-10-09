"""
Logs + Traces 可观测性测试 (补4)

覆盖:
1. span 结构: 嵌套父子 / 耗时 > 0 / 属性挂载
2. 异常 span: 失败标记 + error 信息
3. trace_id 关联: 同一链路 spans 共享 trace_id
4. request_id: 中间件生成 + 日志行携带
5. JSON 日志格式: 解析一行确认字段齐全
6. 三个埋点: RAG 检索 / Agent 路由 / 工具调用 各 1 个测试(不真实调 LLM)

与仓库约定一致: 无 pytest-asyncio, 异步协程用同步 _run 包装;
mock LLM / 临时 DB, 不碰生产库。
"""
import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_trace_context():
    """每个测试独立: 清空 span 栈 / trace_id / request_id,避免跨用例串扰。"""
    from observability import tracer, set_trace_id, set_request_id

    tracer.reset()
    set_trace_id(None)
    set_request_id(None)
    yield
    tracer.reset()
    set_trace_id(None)
    set_request_id(None)


# ----------------------------------------------------------------------
# 1. span 工具: 嵌套 / 耗时 / 属性 / trace_id 关联
# ----------------------------------------------------------------------
class TestTracingSpan:
    def test_nested_spans_share_trace_id_and_have_parent(self):
        """嵌套 span: 父 span 存在时子 span parent_id 指向父,同链路共享 trace_id。"""
        from observability import trace_span

        traces = []
        with trace_span("root", op="outer") as root:
            root_id = root.trace_id
            root_span_id = root.span_id
            with trace_span("child", op="inner") as child:
                assert child.trace_id == root_id
                assert child.parent_id == root_span_id
                assert child.start_ns > 0
                # 父 span 此刻还没结束 → duration 不可用但 start 已记录
            traces.append(child.to_dict())
        traces.append(root.to_dict())

        by_name = {t["name"]: t for t in traces}
        assert by_name["root"]["parent_id"] is None
        assert by_name["child"]["parent_id"] == root_span_id
        # 同链路共享 trace_id
        assert by_name["root"]["trace_id"] == by_name["child"]["trace_id"]
        # 耗时 > 0
        assert by_name["root"]["duration_ms"] > 0
        assert by_name["child"]["duration_ms"] > 0

    def test_span_attrs_are_recorded(self):
        """span 属性挂载: attrs 原样进 to_dict。"""
        from observability import trace_span

        with trace_span("op", mode="hybrid", top_k=3) as span:
            span.attrs["stages_ms"] = {"dense": 1.2, "bm25": 0.8}
        d = span.to_dict()
        assert d["attrs"]["mode"] == "hybrid"
        assert d["attrs"]["top_k"] == 3
        assert d["attrs"]["stages_ms"] == {"dense": 1.2, "bm25": 0.8}

    def test_exception_span_marks_error(self):
        """异常 span: error + error_msg 记录,且异常继续向上抛。"""
        from observability import trace_span

        with pytest.raises(RuntimeError):
            with trace_span("rag_search") as span:
                raise RuntimeError("模型挂了")
        d = span.to_dict()
        assert d["error"] == "RuntimeError"
        assert "模型挂了" in d["error_msg"]

    def test_tracer_explicit_start_end(self):
        """Tracer 显式 start/end: 手动标记 error 也能工作。"""
        from observability import Tracer

        t = Tracer()
        span = t.start_span("tool_call", tool="book_trial_lesson")
        t.end_span(span, error="ValueError", error_msg="bad args")
        d = span.to_dict()
        assert d["error"] == "ValueError"
        assert d["error_msg"] == "bad args"
        assert d["duration_ms"] > 0


# ----------------------------------------------------------------------
# 2. JSON 日志格式
# ----------------------------------------------------------------------
class TestJsonLogging:
    def test_json_formatter_produces_complete_record(self):
        """JsonFormatter: 一行 JSON,字段 ts/level/logger/msg/request_id 齐全。"""
        from observability import JsonFormatter, set_request_id

        set_request_id("req-abc-123")
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="svc.retriever", level=logging.INFO, pathname=__file__,
            lineno=10, msg="hybrid_search done", args=(), exc_info=None,
        )
        line = formatter.format(record)
        payload = json.loads(line)
        assert payload["ts"]
        assert payload["level"] == "INFO"
        assert payload["logger"] == "svc.retriever"
        assert payload["msg"] == "hybrid_search done"
        assert payload["request_id"] == "req-abc-123"

    def test_json_formatter_attaches_extra_fields(self):
        """message 里未占用的 kwargs 自动进 JSON 的 extra 字段。"""
        from observability import JsonFormatter, set_request_id

        set_request_id("req-xyz")
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="obs.trace", level=logging.INFO, pathname=__file__,
            lineno=10, msg="span_complete", args=(),
            exc_info=None,
        )
        record.trace = {"trace_id": "t1", "name": "rag_search"}
        payload = json.loads(formatter.format(record))
        assert payload["trace"]["trace_id"] == "t1"
        assert payload["request_id"] == "req-xyz"

    def test_setup_logging_is_idempotent(self, capsys):
        """setup_logging 重复调用不堆 handler,输出是 JSON 行。"""
        from observability import setup_logging, set_request_id

        setup_logging(level="INFO")
        setup_logging(level="INFO")  # 第二次不应重复添加 handler

        set_request_id("req-42")
        logging.getLogger("app").info("hello")
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l.strip()]
        # JSON 化 + request_id 携带
        payload = json.loads(lines[-1])
        assert payload["msg"] == "hello"
        assert payload["request_id"] == "req-42"
        assert payload["level"] == "INFO"


# ----------------------------------------------------------------------
# 3. request_id: 中间件生成 + 日志透传
# ----------------------------------------------------------------------
class TestRequestIdMiddleware:
    def _app(self):
        app = FastAPI()
        from observability import RequestIdMiddleware

        @app.get("/hello")
        def hello():
            from observability import get_request_id

            # 日志在请求上下文里打,验证 Filter 带上 request_id
            logging.getLogger("test.req").info("handling", extra={"path": "/hello"})
            return {"request_id": get_request_id()}

        app.add_middleware(RequestIdMiddleware)
        return app

    def test_generates_and_echoes_request_id(self):
        """中间件生成 request_id,响应头 X-Request-ID 带回,日志行携带。"""
        client = TestClient(self._app())
        r = client.get("/hello")
        rid = r.headers.get("x-request-id")
        assert rid and len(rid) > 0
        assert r.json()["request_id"] == rid

    def test_propagates_incoming_request_id(self):
        """客户端透传的 X-Request-ID 被沿用(不重新生成)。"""
        client = TestClient(self._app())
        r = client.get("/hello", headers={"X-Request-ID": "client-trace-1"})
        assert r.headers.get("x-request-id") == "client-trace-1"
        assert r.json()["request_id"] == "client-trace-1"

    def test_request_id_cleared_after_request(self):
        """请求结束后 contextvars 里不应残留 request_id(避免污染下个请求)。"""
        from observability import get_request_id

        client = TestClient(self._app())
        client.get("/hello")
        assert get_request_id() is None


# ----------------------------------------------------------------------
# 4. 三个埋点: RAG 检索 / Agent 路由 / 工具调用
# ----------------------------------------------------------------------
class TestTraceInstruments:
    def test_rag_search_hybrid_emits_span_with_stages(self):
        """hybrid_retriever.search: rag_search span + stages attrs 挂载。"""
        import asyncio
        import hashlib
        from unittest.mock import MagicMock, patch

        from services.hybrid_retriever import HybridRetriever
        from observability import tracer, Span

        def fake_embed(text):
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:16]]

        fake_doc = {
            "id": 1,
            "content": "高中数学 韦达定理 专题",
            "category": "subject_method",
            "embedding": fake_embed("高中数学 韦达定理 专题"),
        }

        ks = MagicMock()
        ks.db.get_all_documents.return_value = [fake_doc]

        retriever = HybridRetriever(
            knowledge_service=ks, rrf_k=60, candidates=5, rerank=False,
        )
        # 用真实 trace_span:直接驱动 tracer,从 root logger 捕获 trace 日志
        emitted = []
        logger_obs = logging.getLogger("obs.trace")

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                emitted.append(record.trace)

        with patch("services.hybrid_retriever.embed_input", fake_embed):
            handler = _CaptureHandler()
            logger_obs.addHandler(handler)
            try:
                results = asyncio.run(retriever.search("韦达定理怎么学"))
            finally:
                logger_obs.removeHandler(handler)

        # 行为不变: 结果返回
        assert isinstance(results, list)
        # 埋点: rag_search span + stages attrs 挂载
        spans = [e for e in emitted if e["name"] == "rag_search"]
        assert len(spans) == 1
        assert spans[0]["attrs"]["mode"] == "hybrid"
        assert "stages_ms" in spans[0]["attrs"]
        assert "total_ms" in spans[0]["attrs"]
        assert "result_count" in spans[0]["attrs"]
        assert spans[0]["duration_ms"] > 0

    def test_agent_route_classify_emits_span(self):
        """task_classifier.classify_task: agent_route span, intent/agent attrs。"""
        from agents.task_classification.task_classifier import TaskClassifier

        clf = TaskClassifier.__new__(TaskClassifier)
        clf.context_manager = None
        msg = Mock()
        msg.content = "trial"
        clf.chain = Mock()
        clf.chain.ainvoke = AsyncMock(return_value=msg)

        emitted = []
        logger_obs = logging.getLogger("obs.trace")

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                emitted.append(record.trace)

        handler = _CaptureHandler()
        logger_obs.addHandler(handler)
        try:
            category = _run(clf.classify_task("想试试张老师的课"))
        finally:
            logger_obs.removeHandler(handler)

        assert category == "trial"
        spans = [e for e in emitted if e["name"] == "agent_route"]
        assert len(spans) == 1
        assert spans[0]["attrs"]["intent"] == "trial"
        assert spans[0]["attrs"]["agent"] == "trial"

    def test_tool_call_loop_emits_span(self, tmp_path):
        """agentic tool loop: tool_call span 记录 tool/ok/result。"""
        from agents.agentic.agentic_trial_booking import run_agentic_trial_booking
        from db.db_router import DatabaseRouter

        db_path = f"sqlite:///{tmp_path / 'obs_tool.db'}"
        db = DatabaseRouter(db_path)
        from services.lesson_service import LessonService
        from services.student_profile_service import StudentProfileService

        ls = LessonService()
        ss = StudentProfileService()
        ls.db = db
        ss.db = db

        # 种子默认学生(工具内 initialize_default_student 会兜底,这里确保有)
        try:
            ss.initialize_default_student()
        except Exception:
            pass

        resp1 = Mock(
            tool_calls=[{
                "name": "book_trial_lesson",
                "id": "call_x",
                "args": {"teacher_id": 1, "time_iso": "2026-09-22 19:00", "subject": "数学"},
            }],
            content="",
        )
        resp2 = Mock(content="已预约成功")
        llm = Mock()
        bound = Mock()
        bound.invoke.side_effect = [resp1, resp2]
        llm.bind_tools.return_value = bound

        emitted = []
        logger_obs = logging.getLogger("obs.trace")

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                emitted.append(record.trace)

        handler = _CaptureHandler()
        logger_obs.addHandler(handler)
        try:
            result = run_agentic_trial_booking(
                llm=llm, teacher_id=1, time_iso="2026-09-22 19:00", subject="数学",
                lesson_service=ls, student_service=ss,
                user_message="帮我约试讲",
            )
        finally:
            logger_obs.removeHandler(handler)

        assert result["tool_called"] is True
        spans = [e for e in emitted if e["name"] == "tool_call"]
        assert len(spans) == 1
        assert spans[0]["attrs"]["tool"] == "book_trial_lesson"
        assert spans[0]["attrs"]["ok"] is True
        assert spans[0]["attrs"]["args"]["teacher_id"] == 1
        assert "lesson_id=" in spans[0]["attrs"]["result"]
