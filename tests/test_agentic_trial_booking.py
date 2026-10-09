"""
Agentic 工具调用循环测试
  - @tool book_trial_lesson 定义可导入、参数 schema 正确
  - run_agentic_trial_booking 完整 tool loop(mock LLM + 临时 DB):工具被调、真实落库、返回预约信息
  - 冲突时间 → 工具 ERROR → 喂回 LLM(mock 二次响应)→ 最终 message 是换时间引导
  - LLM 无 tool_call → 返回 tool_called=False
  - /agentic/chat 路由存在并返回结构

约束(与 tests/test_reschedule_routing.py 一致):
  - 无 pytest-asyncio,异步用例统一用 _run 同步包装(本模块实际是同步,无需 asyncio)
  - mock LLM(bind_tools 返回可编程 mock),仓库用临时文件 SQLite(tmp_path),不碰生产库
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agentic.agentic_trial_booking import (  # noqa: E402
    run_agentic_trial_booking,
    _make_book_trial_lesson_tool,
)
from db.db_router import DatabaseRouter  # noqa: E402


# ---------------------------------------------------------------------------
# 临时 DB 服务集:TeacherService / StudentProfileService / LessonService 共用一个
# DatabaseRouter,种子一个老师 + 默认学生,保证 create_lesson 能真实落库。
# ---------------------------------------------------------------------------
def _build_services(tmp_path, teacher_id: int = 1):
    db_path = f"sqlite:///{tmp_path / 'agentic_test.db'}"
    db = DatabaseRouter(db_path)

    from services.lesson_service import LessonService
    from services.student_profile_service import StudentProfileService
    from services.teacher_service import TeacherService

    ls = LessonService()
    ss = StudentProfileService()
    ts = TeacherService()
    ls.db = db
    ss.db = db
    ts.db = db

    # 种子老师(teacher_id 由注入确定),让冲突检查/落库有真实行
    try:
        from db.models import Teacher, Base
        session = db.session_manager.Session()
        existing = session.query(Teacher).filter_by(id=teacher_id).first()
        if not existing:
            t = Teacher(id=teacher_id, name="张明", subjects="数学,物理",
                        grade_levels="初中,高中", teaching_style="严格",
                        hourly_rate=350.0)
            session.add(t)
            session.commit()
        session.close()
    except Exception:
        # 种子失败不阻断:create_lesson 不一定依赖 teacher 行(FK 未强制)
        pass

    # 确保默认学员存在
    try:
        ss.initialize_default_student()
    except Exception:
        pass

    return ls, ss


def _lesson_count(ls) -> int:
    session = ls.db.session_manager.Session()
    from db.models import Lesson
    try:
        return session.query(Lesson).count()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. tool 定义与 schema
# ---------------------------------------------------------------------------
class TestToolDefinition:
    def test_tool_importable_and_schema(self):
        tool_fn = _make_book_trial_lesson_tool()
        schema = tool_fn.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "teacher_id" in props
        assert "time_iso" in props
        assert "subject" in props
        assert "duration_minutes" in props
        # teacher_id/time_iso 必填,subject/duration 有默认
        req = schema.get("required", [])
        assert "teacher_id" in req
        assert "time_iso" in req
        assert "duration_minutes" not in req
        assert "subject" not in req

    def test_tool_description_mentions_time_format(self):
        tool_fn = _make_book_trial_lesson_tool()
        desc = (tool_fn.description or "")
        assert "teacher_id" in desc
        assert "time_iso" in desc
        assert "YYYY-MM-DD" in desc
        assert "OK: lesson_id=" in desc
        assert "ERROR" in desc


# ---------------------------------------------------------------------------
# 2. 完整 tool loop(成功 + 落库 + 预约信息)
# ---------------------------------------------------------------------------
class TestToolLoopSuccess:
    def _bound_llm(self, resp1, resp2):
        llm = Mock()
        bound = Mock()
        bound.invoke.side_effect = [resp1, resp2]
        llm.bind_tools.return_value = bound
        return llm

    def test_tool_called_and_persisted(self, tmp_path):
        ls, ss = _build_services(tmp_path, teacher_id=7)
        before = _lesson_count(ls)
        time_iso = "2026-09-20 19:00"

        llm = self._bound_llm(
            Mock(tool_calls=[{
                "name": "book_trial_lesson",
                "id": "call_1",
                "args": {"teacher_id": 7, "time_iso": time_iso, "subject": "数学"},
            }], content=""),
            Mock(content="已为您预约张明老师 2026-09-20 19:00 的免费试讲，请准时参加！"),
        )

        result = run_agentic_trial_booking(
            llm=llm, teacher_id=7, time_iso=time_iso, subject="数学",
            lesson_service=ls, student_service=ss,
            user_message="预约 2026-09-20 19:00 的试讲",
        )

        assert result["tool_called"] is True
        assert isinstance(result["lesson_id"], int)
        assert result["lesson_id"] > 0
        assert "张明老师" in result["message"]
        assert _lesson_count(ls) == before + 1  # 真实落库

    def test_conflict_returns_change_time_guide(self, tmp_path):
        ls, ss = _build_services(tmp_path, teacher_id=8)
        # 先占住该时段
        conflict_time = "2026-09-21 20:00"
        _preoccupy(ls, ss, teacher_id=8, time_iso=conflict_time)

        llm = self._bound_llm(
            Mock(tool_calls=[{
                "name": "book_trial_lesson",
                "id": "call_2",
                "args": {"teacher_id": 8, "time_iso": conflict_time},
            }], content=""),
            Mock(content="抱歉，张明老师在该时段已有课程，麻烦您换一个时间预约。"),
        )

        result = run_agentic_trial_booking(
            llm=llm, teacher_id=8, time_iso=conflict_time,
            lesson_service=ls, student_service=ss,
            user_message=f"预约 {conflict_time} 的试讲",
        )

        assert result["tool_called"] is True
        assert "换" in result["message"]
        # ERROR 已喂回 LLM 二次响应
        assert "已有课程" in result["message"]

    def test_no_tool_call_returns_false(self, tmp_path):
        ls, ss = _build_services(tmp_path, teacher_id=9)
        llm = self._bound_llm(
            Mock(tool_calls=[], content="请问您想预约哪位老师、什么时间呢？"),
            Mock(),
        )

        result = run_agentic_trial_booking(
            llm=llm, teacher_id=None, time_iso=None,
            lesson_service=ls, student_service=ss,
            user_message="我想给孩子补数学",
        )

        assert result["tool_called"] is False
        assert result["lesson_id"] is None
        assert "老师" in result["message"]


def _preoccupy(ls, ss, teacher_id: int, time_iso: str):
    """往临时 DB 塞一条已占用该时段的 lesson(直接走 create_lesson)。"""
    dt = datetime.fromisoformat(time_iso.replace("Z", "+00:00"))
    dt = dt.replace(tzinfo=None)
    student = ss.initialize_default_student()
    ok, _ = ls.create_lesson(
        teacher_id=teacher_id, student_id=student["id"], scheduled_at=dt,
        lesson_type="regular", duration_minutes=30, status="confirmed", price=100.0,
    )
    assert ok


# ---------------------------------------------------------------------------
# 3. /agentic/chat 路由存在并返回结构
# ---------------------------------------------------------------------------
class TestAgenticChatRoute:
    def test_route_exists(self):
        from api import agentic as agentic_module
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(agentic_module.router)
        paths = {r.path for r in app.routes}
        assert "/agentic/chat" in paths

    def test_endpoint_returns_structure(self, monkeypatch, tmp_path):
        """mock LLM + run_agentic_trial_booking,避免真实调用,验证返回结构。"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import api.agentic as agentic_module

        app = FastAPI()
        app.include_router(agentic_module.router)

        expected = {"tool_called": True, "lesson_id": 42, "message": "已为您预约"}

        # 让 endpoint 的函数级 import 拿到 stub(不连真实 LLM / 真实 DB)
        import config.model_provider as mp_mod
        import agents.agentic.agentic_trial_booking as atb_mod

        monkeypatch.setattr(mp_mod, "create_chat_model", lambda: Mock())
        monkeypatch.setattr(
            atb_mod, "run_agentic_trial_booking",
            lambda **kw: dict(expected),
        )
        monkeypatch.setattr(
            agentic_module, "_build_services",
            lambda: (Mock(), Mock()),
        )

        client = TestClient(app)
        resp = client.post("/agentic/chat", json={
            "message": "预约明天晚上7点的试讲",
            "teacher_id": 1,
            "time_iso": "2026-09-17 19:00",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["tool_called"] is True
        assert body["lesson_id"] == 42
        assert "已为您预约" in body["message"]