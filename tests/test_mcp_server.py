"""MCP(stdio)server 工具接口单元测试。

覆盖:
1. 工具注册名 / description / input_schema 齐全
2. 5 个只读工具返回结构(ok=true + 业务字段)
3. 写入工具默认被拒(未开 MCP_ALLOW_WRITE)
4. MCP_ALLOW_WRITE=1 但 write_token 错 → 仍拒
5. token 正确 + 身份已声明 → 走通 service 替身(lesson_service 假实现)
6. 身份闸门:未声明身份 / 非本身份学员 → 拒
7. 未知工具名 / 缺参 → 协议层报错(不裸抛到调用方进程外)

**不起真子进程**(直接调用注册的 tool 函数,async 用 asyncio.run);真 stdio 走
scripts/demo_mcp_server.py。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server import server as mcp_server  # noqa: E402

READ_TOOLS = {
    "find_teachers",
    "get_teacher_availability",
    "search_knowledge",
    "list_packages",
    "get_student_lessons",
}
WRITE_TOOLS = {"book_trial_lesson"}


# ----------------------------------------------------------------------
# 替身(不连生产库、不发网络请求)
# ----------------------------------------------------------------------


class _FakeIdentityRepo:
    def __init__(self, issued: set[str], owners: dict[str, str]):
        self.issued = issued
        self.owners = owners

    def identity_exists(self, user_id: str) -> bool:
        return user_id in self.issued


class _FakeIdentity:
    """只实现工具用到的两个方法:identity_exists 走 repo,认领/校验走内存表。"""

    def __init__(self, owners: dict[str, str] | None = None, issued: set[str] | None = None):
        self.owners = dict(owners or {})
        self.repo = _FakeIdentityRepo(set(issued or ()), self.owners)
        self.claimed: list[tuple[str, str]] = []

    def claim_student(self, identity, student_id: str) -> None:
        owner = self.owners.get(student_id)
        if owner is not None and owner != identity.user_id:
            raise mcp_server.IdentityError(f"student {student_id} 已被其它身份认领")
        self.owners[student_id] = identity.user_id
        self.claimed.append((identity.user_id, student_id))

    def assert_student_owned(self, identity, student_id: str) -> str:
        owner = self.owners.get(student_id)
        if owner is None:
            raise mcp_server.IdentityError(f"student {student_id} 尚未认领,不可访问其业务数据")
        if owner != identity.user_id:
            raise mcp_server.IdentityError(f"student {student_id} 不属于当前身份")
        return student_id


class _FakeTeacher:
    def __init__(self):
        self.calls: list[tuple] = []

    def get_available_teachers(self):
        self.calls.append(("available",))
        return [{"id": 9, "name": "李四", "subjects": ["英语"], "grade_levels": ["高中"],
                 "hourly_rate": 300.0, "rating": 4.8}]

    def find_by_subject(self, subject):
        self.calls.append(("subject", subject))
        return [{"id": 1, "name": "张明", "subjects": [subject], "grade_levels": ["高中"],
                 "hourly_rate": 350.0, "rating": 4.9}]

    def find_by_subject_and_grade(self, subject, grade_level):
        self.calls.append(("subject_grade", subject, grade_level))
        return [{"id": 2, "name": "王五", "subjects": [subject],
                 "grade_levels": [grade_level], "hourly_rate": 320.0, "rating": 4.7}]

    def get_teacher_availability(self, teacher_id):
        self.calls.append(("availability", teacher_id))
        return [{"day": "monday", "slots": ["19:00-20:00"]}]


class _FakeKnowledge:
    def __init__(self):
        self.initialized = False
        self.queries: list[tuple] = []

    async def initialize(self):
        self.initialized = True

    async def search(self, query, top_k=3):
        self.queries.append((query, top_k))
        return [{"id": 7, "category": "course_intro", "subject": "数学",
                 "grade_level": "高中", "score": 0.87, "rank": 1,
                 "content": "高中数学 12 周提分方案"}]


class _FakePackage:
    def get_all_packages(self):
        return [{"id": 1, "name": "高一数学 12 周提分套餐", "teacher_id": 1,
                 "subject": "数学", "grade_level": "高中", "total_lessons": 24,
                 "duration_minutes": 60, "price_total": 7200.0,
                 "price_per_lesson": 300.0, "validity_days": 120,
                 "description": "函数+几何双模块"}]


class _FakeLesson:
    def __init__(self, success: bool = True, result=101):
        self.success = success
        self.result = result
        self.created: list[dict] = []

    def get_student_upcoming_lessons(self, student_id, limit=5):
        return [{"id": 55, "teacher_id": 1, "student_id": student_id,
                 "scheduled_at": "2026-09-20 19:00:00", "duration_minutes": 60,
                 "lesson_type": "regular", "status": "confirmed", "subject": "数学"}]

    def create_lesson(self, **kwargs):
        self.created.append(kwargs)
        return self.success, self.result


def _services(owners=None, issued=None, lesson=None) -> mcp_server.Services:
    return mcp_server.Services(
        teacher=_FakeTeacher(),
        knowledge=_FakeKnowledge(),
        package=_FakePackage(),
        lesson=lesson or _FakeLesson(),
        identity=_FakeIdentity(owners=owners, issued=issued),
    )


ANON = "anon_TESTCALLER000000000000"
OTHER = "anon_OTHERIDENTITY000000000"

# student namespace 在身份体系里是字符串;业务表 lessons.student_id 是整数 FK,
# 因此对外约定用数字字符串(既满足身份 namespace,又能直通业务查询)。
STUDENT = "1"


@pytest.fixture()
def svc() -> mcp_server.Services:
    return _services(owners={STUDENT: ANON}, issued={ANON, OTHER})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """默认清空三个 MCP env,避免外部环境串味;用例内自行开启。"""
    for name in (mcp_server.ALLOW_WRITE_ENV, mcp_server.WRITE_TOKEN_ENV,
                 mcp_server.CALLER_USER_ID_ENV):
        monkeypatch.delenv(name, raising=False)


def _loads(payload: str) -> dict:
    return json.loads(payload)


# ----------------------------------------------------------------------
# 1. 注册名 / schema
# ----------------------------------------------------------------------


def test_all_tools_registered_with_name_description_and_schema():
    mcp = mcp_server.create_server(service_factory=lambda: _services())
    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}

    assert set(by_name) == READ_TOOLS | WRITE_TOOLS
    for name, tool in by_name.items():
        assert tool.description, f"{name} 缺少 description(中文用途说明)"
        assert tool.input_schema.get("type") == "object", f"{name} input_schema 不是 object"


def test_read_tool_schemas_declare_expected_properties():
    mcp = mcp_server.create_server(service_factory=lambda: _services())
    by_name = {t.name: t for t in asyncio.run(mcp.list_tools())}

    assert set(by_name["find_teachers"].input_schema["properties"]) == {"subject", "grade_level"}
    assert by_name["find_teachers"].input_schema["required"] == ["subject"]
    assert set(by_name["get_teacher_availability"].input_schema["properties"]) == {"teacher_id"}
    assert set(by_name["search_knowledge"].input_schema["properties"]) == {"query", "top_k"}
    assert by_name["list_packages"].input_schema["properties"] == {}
    assert set(by_name["get_student_lessons"].input_schema["properties"]) == {"student_id", "limit"}
    assert set(by_name["book_trial_lesson"].input_schema["properties"]) == {
        "teacher_id", "student_id", "scheduled_at", "subject", "write_token",
    }


# ----------------------------------------------------------------------
# 2. 只读工具返回结构
# ----------------------------------------------------------------------


def test_find_teachers_returns_structured_json(svc):
    result = _loads(mcp_server.find_teachers(svc, "数学", "高中"))
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["query"] == {"subject": "数学", "grade_level": "高中"}
    assert result["teachers"][0]["name"] == "王五"
    assert ("subject_grade", "数学", "高中") in svc.teacher.calls


def test_find_teachers_without_grade_uses_subject_lookup(svc):
    result = _loads(mcp_server.find_teachers(svc, "英语"))
    assert result["ok"] is True
    assert result["teachers"][0]["name"] == "张明"
    assert ("subject", "英语") in svc.teacher.calls


def test_get_teacher_availability_and_list_packages(svc):
    slots = _loads(mcp_server.get_teacher_availability(svc, 1))
    assert slots["ok"] is True and slots["count"] == 1
    assert slots["availability"][0]["day"] == "monday"

    packages = _loads(mcp_server.list_packages(svc))
    assert packages["ok"] is True and packages["count"] == 1
    assert packages["packages"][0]["name"] == "高一数学 12 周提分套餐"


def test_search_knowledge_initializes_and_returns_documents(svc):
    result = _loads(asyncio.run(mcp_server.search_knowledge(svc, "高中提分", top_k=2)))
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["documents"][0]["content"] == "高中数学 12 周提分方案"
    assert svc.knowledge.initialized is True
    assert svc.knowledge.queries == [("高中提分", 2)]


def test_search_knowledge_rejects_empty_query(svc):
    result = _loads(asyncio.run(mcp_server.search_knowledge(svc, "   ")))
    assert result == {"ok": False, "error": "missing_query", "detail": "query 不能为空"}


def test_get_student_lessons_requires_declared_identity(svc, monkeypatch):
    result = _loads(mcp_server.get_student_lessons(svc, STUDENT))
    assert result["ok"] is False
    assert result["error"] == "identity_required"


def test_get_student_lessons_denies_foreign_student(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, OTHER)
    result = _loads(mcp_server.get_student_lessons(svc, STUDENT))
    assert result["ok"] is False
    assert result["error"] == "student_not_owned"


def test_get_student_lessons_returns_rows_for_owner(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)
    result = _loads(mcp_server.get_student_lessons(svc, STUDENT, limit=3))
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["lessons"][0]["id"] == 55


@pytest.mark.parametrize("legacy_id", ["default_user"])
def test_get_student_lessons_rejects_legacy_namespace(svc, monkeypatch, legacy_id):
    """default_user 是 legacy 内部命名空间;即使它出现在 identities 表也不得被外部用作身份。"""
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, legacy_id)
    result = _loads(mcp_server.get_student_lessons(svc, legacy_id))
    assert result["ok"] is False
    assert result["error"] == "identity_required"


def test_caller_user_id_must_be_issued_by_server(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, "anon_NOT_ISSUED_0000000000000")
    result = _loads(mcp_server.get_student_lessons(svc, STUDENT))
    assert result["ok"] is False
    assert result["error"] == "identity_required"


# ----------------------------------------------------------------------
# 3. 写入闸门:默认关闭
# ----------------------------------------------------------------------


def test_book_trial_lesson_denied_by_default(svc):
    result = _loads(mcp_server.book_trial_lesson(svc, 1, STUDENT, "2026-09-20 19:00"))
    assert result["ok"] is False
    assert result["error"] == "write_disabled"
    assert svc.lesson.created == []


def test_book_trial_lesson_denied_when_token_missing_in_env(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    result = _loads(mcp_server.book_trial_lesson(svc, 1, STUDENT, "2026-09-20 19:00",
                                                 write_token="anything"))
    assert result["ok"] is False
    assert result["error"] == "write_disabled"


def test_book_trial_lesson_denied_on_wrong_token(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    result = _loads(mcp_server.book_trial_lesson(svc, 1, STUDENT, "2026-09-20 19:00",
                                                 write_token="wrong-token"))
    assert result["ok"] is False
    assert result["error"] == "write_disabled"
    assert svc.lesson.created == []


def test_book_trial_lesson_denied_without_declared_identity(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    result = _loads(mcp_server.book_trial_lesson(svc, 1, STUDENT, "2026-09-20 19:00",
                                                 write_token="secret-token"))
    assert result["ok"] is False
    assert result["error"] == "identity_required"


# ----------------------------------------------------------------------
# 4. 写入闸门:token 正确 + 身份声明 → 走通 service 替身
# ----------------------------------------------------------------------


def test_book_trial_lesson_succeeds_with_token_and_identity(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        svc, 1, STUDENT, "2026-09-20 19:00", subject="数学", write_token="secret-token",
    ))
    assert result["ok"] is True
    assert result["lesson_id"] == 101

    assert len(svc.lesson.created) == 1
    created = svc.lesson.created[0]
    assert created["lesson_type"] == "trial"
    assert created["duration_minutes"] == mcp_server.TRIAL_DURATION_MINUTES
    assert created["status"] == "confirmed"
    assert created["subject"] == "数学"
    assert created["scheduled_at"].hour == 19
    # 身份闸门:先认领到调用方身份
    assert svc.identity.claimed == [(ANON, STUDENT)]


def test_book_trial_lesson_claims_new_student_for_caller(monkeypatch):
    """未被任何身份认领的 student namespace → 认领给调用方后落库。"""
    service = _services(owners={}, issued={ANON})
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        service, 1, "77", "2026-09-20T19:00:00", write_token="secret-token",
    ))
    assert result["ok"] is True
    assert service.identity.owners["77"] == ANON


def test_book_trial_lesson_denies_student_owned_by_other_identity(monkeypatch):
    service = _services(owners={STUDENT: OTHER}, issued={ANON, OTHER})
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        service, 1, STUDENT, "2026-09-20 19:00", write_token="secret-token",
    ))
    assert result["ok"] is False
    assert result["error"] == "student_not_owned"
    assert service.lesson.created == []


def test_book_trial_lesson_rejects_legacy_student_namespace(monkeypatch):
    service = _services(owners={}, issued={ANON})
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        service, 1, "default_user", "2026-09-20 19:00", write_token="secret-token",
    ))
    assert result["ok"] is False
    assert result["error"] == "invalid_student_id"


def test_book_trial_lesson_surfaces_service_rejection(monkeypatch):
    lesson = _FakeLesson(success=False, result="老师在该时段已有课程，请选择其他时间")
    service = _services(owners={STUDENT: ANON}, issued={ANON}, lesson=lesson)
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        service, 1, STUDENT, "2026-09-20 19:00", write_token="secret-token",
    ))
    assert result["ok"] is False
    assert result["error"] == "lesson_create_rejected"


def test_book_trial_lesson_rejects_bad_datetime(svc, monkeypatch):
    monkeypatch.setenv(mcp_server.ALLOW_WRITE_ENV, "1")
    monkeypatch.setenv(mcp_server.WRITE_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(mcp_server.CALLER_USER_ID_ENV, ANON)

    result = _loads(mcp_server.book_trial_lesson(
        svc, 1, STUDENT, "九点以后", write_token="secret-token",
    ))
    assert result["ok"] is False
    assert result["error"] == "invalid_scheduled_at"


# ----------------------------------------------------------------------
# 5. 错误不外泄 / 协议层行为
# ----------------------------------------------------------------------


def test_read_tool_failure_is_structured_not_raised():
    class _BoomTeacher(_FakeTeacher):
        def get_available_teachers(self):
            raise RuntimeError("db down")

    service = _services()
    service = mcp_server.Services(teacher=_BoomTeacher(), knowledge=service.knowledge,
                                  package=service.package, lesson=service.lesson,
                                  identity=service.identity)
    result = _loads(mcp_server.find_teachers(service, ""))
    assert result["ok"] is False
    assert result["error"] == "teacher_query_failed"


def test_services_constructed_lazily_and_reused():
    built: list[int] = []

    def factory():
        built.append(1)
        return _services()

    mcp = mcp_server.create_server(service_factory=factory)
    tool = mcp._tool_manager.get_tool("list_packages")
    assert built == []  # 注册期不起 IO
    tool.fn()
    tool.fn()
    assert len(built) == 1


def test_unknown_tool_name_raises_at_protocol_layer():
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = mcp_server.create_server(service_factory=lambda: _services())
    with pytest.raises(ToolError, match="Unknown tool: not_a_tool"):
        asyncio.run(mcp.call_tool("not_a_tool", {}))


def test_missing_required_argument_raises_at_protocol_layer():
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = mcp_server.create_server(service_factory=lambda: _services())
    with pytest.raises(ToolError, match="validation error"):
        asyncio.run(mcp.call_tool("find_teachers", {}))


def test_registered_tool_function_returns_json_string():
    """注册的函数被协议层调用时返回 json 字符串(非 dict),保证 MCP 语义一致。"""
    mcp = mcp_server.create_server(service_factory=lambda: _services())
    result = asyncio.run(mcp.call_tool("list_packages", {}))
    assert result.content[0].text.startswith('{"ok": true')
