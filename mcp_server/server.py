"""MCP(stdio)server:把已有 service 能力暴露给外部 Agent。

设计边界(重要,勿夸大):
- 本系统**没有登录系统**(见 config/identity.py:8-9),user_id 只是服务端签发的
  匿名身份(anon_*),**不能证明**真实家长身份。因此:
  - 只读工具(老师匹配 / 档期 / 知识检索 / 套餐 / 学员课时)默认开放;
  - 写入工具(试讲预约)默认**拒绝** —— 仅当 env `MCP_ALLOW_WRITE=1` **且**调用方
    传入的 `write_token` 与 env `MCP_WRITE_TOKEN` 完全一致时才执行,对应
    "匿名只开放咨询/公开老师匹配,预约需身份验证"。
- 只读学员数据也必须过身份闸门:匿名身份必须先被调用方以服务端签发过的
  `MCP_CALLER_USER_ID` 声明,且该学员 namespace 已归属该身份,否则拒绝。
  **禁止**使用 LEGACY_DEFAULT_USER_ID("default_user")命名空间 —— 那是 legacy
  内部路径专用,暴露给外部 MCP 等于让外部读/写他人数据。
- 本模块是并列新增:不接入 /chat/stream,不改 agents/agentic 的 bind_tools 主链路。
  依赖方向单向:mcp_server/ → services/ / config/ / db/。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

from mcp.server import MCPServer

from config.identity import LEGACY_DEFAULT_USER_ID, Identity
from config.settings import settings
from db.db_router import DatabaseRouter
from services.identity_service import IdentityError, IdentityService
from services.knowledge_service import KnowledgeService
from services.lesson_service import LessonService
from services.package_service import PackageService
from services.teacher_service import TeacherService

# ----------------------------------------------------------------------
# 常量(魔法值集中于此)
# ----------------------------------------------------------------------

MCP_SERVER_NAME = "smart-tutor"
MCP_SERVER_VERSION = "0.1.0"

# 写入闸门:两个环境变量必须同时满足
ALLOW_WRITE_ENV = "MCP_ALLOW_WRITE"
WRITE_TOKEN_ENV = "MCP_WRITE_TOKEN"
# 调用方声明自己是哪个**服务端签发过的**匿名身份(get_student_lessons / book_trial_lesson 用)
CALLER_USER_ID_ENV = "MCP_CALLER_USER_ID"

# 试讲时长与落库参数(与 agents/agentic/agentic_trial_booking.py 的试讲路径保持一致)
TRIAL_DURATION_MINUTES = 30
TRIAL_PRICE = 0.0
TRIAL_LESSON_TYPE = "trial"
TRIAL_LESSON_STATUS = "confirmed"

# 匿名身份前缀:只接受服务端签发形态,拒绝任意字符串冒充
ANON_USER_ID_PREFIX = "anon_"

# 工具返回值里 content 截断长度(避免外部 Agent 上下文被长文档撑爆)
SNIPPET_CHARS = 400


# ----------------------------------------------------------------------
# 工具实现(纯函数式,首参注入 services,便于测试直接调用 → 不起子进程)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Services:
    """server 用到的 service 集合(frozen:工具内不得替换实现)。"""

    teacher: Any
    knowledge: Any
    package: Any
    lesson: Any
    identity: Any


def _build_service(cls: type, url: str) -> Any:
    """构造 TeacherService / PackageService / LessonService 并把库路径换成 url。

    这三个服务的 __init__ 把库路径硬编码为默认相对路径 `sqlite:///data/smart_appointment.db`
    且不接受 db_path 参数。直接实例化会先对生产库跑一次 create_all + 迁移,违背
    "DATABASE_URL 指向临时库" 的隔离要求;因此跳过 __init__(它们只做 self.db 赋值),
    直接注入目标 DatabaseRouter。
    """
    instance = cls.__new__(cls)
    instance.db = DatabaseRouter(url)
    return instance


def build_services(db_path: Optional[str] = None) -> Services:
    """按配置库路径构造 service 集合(DATABASE_URL / 显式 db_path 真正生效)。"""
    url = db_path or settings.database_url
    return Services(
        teacher=_build_service(TeacherService, url),
        knowledge=KnowledgeService(url),
        package=_build_service(PackageService, url),
        lesson=_build_service(LessonService, url),
        identity=IdentityService(url),
    )


def _ok(**payload: Any) -> str:
    """成功返回:{ok: true, ...}。统一 json.dumps(ensure_ascii=False) 字符串。"""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, default=str)


def _err(error: str, **extra: Any) -> str:
    """失败返回结构化 JSON:{"ok": false, "error": "..."};不抛裸异常出协议层。"""
    return json.dumps({"ok": False, "error": error, **extra}, ensure_ascii=False, default=str)


def _env_flag(name: str) -> bool:
    """env 开关:仅 "1" / "true" / "yes" / "on" 视为开启(默认关闭)。"""
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _caller_identity(services: Services) -> Optional[Identity]:
    """解析调用方匿名身份。

    只接受**服务端已签发**的 anon_* user_id(identities 表命中),否则返回 None。
    绝不接受 LEGACY_DEFAULT_USER_ID:那是 legacy 内部路径命名空间,暴露给外部 MCP
    等于让任何调用方读写 default_user 的数据。
    """
    raw = (os.getenv(CALLER_USER_ID_ENV) or "").strip()
    if not raw or raw == LEGACY_DEFAULT_USER_ID:
        return None
    if not raw.startswith(ANON_USER_ID_PREFIX):
        return None
    if not services.identity.repo.identity_exists(raw):
        return None
    return Identity(user_id=raw, legacy=False)


def _coerce_student_id(raw: Any) -> Tuple[Optional[int], Optional[str]]:
    """MCP 入参 student_id 是字符串;业务表 lessons.student_id 是整数 FK。"""
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None, "缺少 student_id"
    try:
        return int(text), None
    except (TypeError, ValueError):
        return None, f"student_id 必须是数字字符串,收到 {raw!r}"


def _parse_scheduled_at(raw: Any) -> Tuple[Optional[datetime], Optional[str]]:
    """解析 `YYYY-MM-DD HH:MM` 或 ISO 8601,按无时区(本地)语义返回。"""
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None, "缺少 scheduled_at"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None, f"无法解析 scheduled_at {raw!r},请用 YYYY-MM-DD HH:MM 格式"
    # 与既有预约路径一致:统一 strip tzinfo 后按本地时间比较/落库
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed, None


def _truncate(text: Any, limit: int = SNIPPET_CHARS) -> str:
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[:limit] + "…"


# --------------------------- 只读工具实现 ---------------------------


def find_teachers(services: Services, subject: str, grade_level: Optional[str] = None) -> str:
    """按学科(+可选年级段)匹配老师。

    - 传 grade_level:走 find_by_subject_and_grade(学科 + 年级硬约束)
    - 不传 grade_level:走 find_by_subject(仅学科)
    - subject 为空:退化为 get_available_teachers(全部可约老师)
    返回每位老师的 id / 姓名 / 学科 / 年级段 / 时薪 / 评分 / 简介摘要。
    """
    subject = (subject or "").strip()
    grade = (grade_level or "").strip() or None
    try:
        if not subject:
            rows = services.teacher.get_available_teachers()
        elif grade:
            rows = services.teacher.find_by_subject_and_grade(subject, grade)
        else:
            rows = services.teacher.find_by_subject(subject)
    except Exception as exc:  # noqa: BLE001 - 统一转结构化错误,不炸协议层
        return _err("teacher_query_failed", detail=str(exc))
    teachers = [
        {
            "id": row.get("id"),
            "name": row.get("name"),
            "subjects": row.get("subjects"),
            "grade_levels": row.get("grade_levels"),
            "hourly_rate": row.get("hourly_rate"),
            "rating": row.get("rating"),
            "years_experience": row.get("years_experience"),
            "trial_available": row.get("trial_available"),
            "bio": _truncate(row.get("bio")),
        }
        for row in rows or []
    ]
    return _ok(query={"subject": subject or None, "grade_level": grade},
               count=len(teachers), teachers=teachers)


def get_teacher_availability(services: Services, teacher_id: int) -> str:
    """查询指定老师的可约档期(weekday + 时间段列表)。"""
    try:
        teacher_id = int(teacher_id)
    except (TypeError, ValueError):
        return _err("invalid_teacher_id", detail=f"teacher_id 必须是整数,收到 {teacher_id!r}")
    try:
        slots = services.teacher.get_teacher_availability(teacher_id)
    except Exception as exc:  # noqa: BLE001
        return _err("availability_query_failed", detail=str(exc))
    return _ok(teacher_id=teacher_id, count=len(slots or []), availability=slots or [])


async def search_knowledge(services: Services, query: str, top_k: int = 3) -> str:
    """检索知识库(走既有 hybrid / dense 链路)。

    知识库未初始化时先 initialize();检索失败按既有降级语义返回空召回,
    但仍以 ok=true + count=0 返回,由调用方决定话术。
    """
    query = (query or "").strip()
    if not query:
        return _err("missing_query", detail="query 不能为空")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return _err("invalid_top_k", detail=f"top_k 必须是整数,收到 {top_k!r}")
    if top_k <= 0:
        return _err("invalid_top_k", detail="top_k 必须为正整数")

    knowledge = services.knowledge
    try:
        if not getattr(knowledge, "initialized", False):
            await knowledge.initialize()
        docs = await knowledge.search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return _err("knowledge_search_failed", detail=str(exc))
    results = [
        {
            "id": doc.get("id"),
            "category": doc.get("category"),
            "subject": doc.get("subject"),
            "grade_level": doc.get("grade_level"),
            "score": doc.get("score"),
            "rank": doc.get("rank"),
            "content": _truncate(doc.get("content")),
        }
        for doc in docs or []
    ]
    return _ok(query=query, top_k=top_k, count=len(results), documents=results)


def list_packages(services: Services) -> str:
    """列出在售课程套餐(仅 active)。"""
    try:
        rows = services.package.get_all_packages()
    except Exception as exc:  # noqa: BLE001
        return _err("package_query_failed", detail=str(exc))
    packages = [
        {
            "id": row.get("id"),
            "name": row.get("name"),
            "teacher_id": row.get("teacher_id"),
            "subject": row.get("subject"),
            "grade_level": row.get("grade_level"),
            "total_lessons": row.get("total_lessons"),
            "duration_minutes": row.get("duration_minutes"),
            "price_total": row.get("price_total"),
            "price_per_lesson": row.get("price_per_lesson"),
            "validity_days": row.get("validity_days"),
            "description": _truncate(row.get("description")),
        }
        for row in rows or []
    ]
    return _ok(count=len(packages), packages=packages)


def get_student_lessons(services: Services, student_id: str, limit: int = 5) -> str:
    """读取某学员即将到来的已确认课时。

    身份闸门(先于业务查询):
    1. 调用方必须声明服务端签发过的匿名身份(env MCP_CALLER_USER_ID);
    2. 该 student namespace 必须已归属该身份(assert_student_owned),否则 403 语义拒绝。
    本项目无登录系统,匿名身份不足以证明家长真实身份,因此未声明身份一律拒绝。
    """
    identity = _caller_identity(services)
    if identity is None:
        return _err(
            "identity_required",
            detail=(
                "外部调用必须通过环境变量 MCP_CALLER_USER_ID 声明一个服务端已签发的 "
                "anon_* 匿名身份;本项目无登录系统,不接受未声明的身份"
            ),
        )

    numeric_id, error = _coerce_student_id(student_id)
    if error:
        return _err("invalid_student_id", detail=error)
    try:
        # student namespace 就是业务 student_id 的字符串形态(数字字符串),
        # 保证"读的 namespace"与"读的课程数据"指向同一学员,不给越权留缝。
        services.identity.assert_student_owned(identity, str(numeric_id))
    except IdentityError as exc:
        return _err("student_not_owned", detail=exc.message, status_code=exc.status_code)
    except Exception as exc:  # noqa: BLE001
        return _err("identity_check_failed", detail=str(exc))

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _err("invalid_limit", detail=f"limit 必须是整数,收到 {limit!r}")
    if limit <= 0:
        return _err("invalid_limit", detail="limit 必须为正整数")

    try:
        lessons = services.lesson.get_student_upcoming_lessons(numeric_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return _err("lesson_query_failed", detail=str(exc))
    rows = [
        {
            "id": row.get("id"),
            "teacher_id": row.get("teacher_id"),
            "student_id": row.get("student_id"),
            "scheduled_at": row.get("scheduled_at"),
            "duration_minutes": row.get("duration_minutes"),
            "lesson_type": row.get("lesson_type"),
            "status": row.get("status"),
            "subject": row.get("subject"),
        }
        for row in lessons or []
    ]
    return _ok(student_id=numeric_id, count=len(rows), lessons=rows)


def book_trial_lesson(
    services: Services,
    teacher_id: int,
    student_id: str,
    scheduled_at: str,
    subject: Optional[str] = None,
    write_token: str = "",
) -> str:
    """预约一次免费试讲课(**默认拒绝**)。

    闸门顺序:
    1. env MCP_ALLOW_WRITE 必须为真,且 write_token 与 env MCP_WRITE_TOKEN 完全一致
       (MCP_WRITE_TOKEN 为空视为未配置 → 一律拒绝);
    2. 调用方必须声明服务端已签发的匿名身份(MCP_CALLER_USER_ID);
    3. student_id 必须经 IdentityService.claim_student 校验归属 —— 未认领可首次认领,
       已被其它身份认领则 409 拒绝。禁止使用 LEGACY_DEFAULT_USER_ID 命名空间。

    理由:项目无登录系统(config/identity.py:8),匿名身份不足以证明家长真实身份,
    所以外部写入默认拒绝;即使放开,也只能落在调用方自己已认领的学员 namespace 下。
    """
    token = (os.getenv(WRITE_TOKEN_ENV) or "")
    if not _env_flag(ALLOW_WRITE_ENV) or not token or write_token != token:
        return _err(
            "write_disabled",
            detail=(
                "写入工具默认关闭:需 MCP_ALLOW_WRITE=1 且 write_token 与 "
                "MCP_WRITE_TOKEN 完全一致;本项目无登录系统,匿名身份不足以证明家长"
                "真实身份,预约需身份验证"
            ),
        )

    identity = _caller_identity(services)
    if identity is None:
        return _err("identity_required",
                    detail="写入需先通过 MCP_CALLER_USER_ID 声明服务端已签发的匿名身份")

    raw_key = str(student_id).strip() if student_id is not None else ""
    if not raw_key or raw_key == LEGACY_DEFAULT_USER_ID:
        return _err("invalid_student_id",
                    detail="禁止使用 default_user 等 legacy 命名空间,请传真实学员 namespace")
    numeric_id, error = _coerce_student_id(raw_key)
    if error:
        return _err("invalid_student_id", detail=error)
    try:
        # 未认领 → 认领到当前身份;已被他人认领 → 409。namespace 用数字字符串,
        # 与落库的 student_id 一致,避免"认领 A、写入 B"的错位。
        services.identity.claim_student(identity, str(numeric_id))
    except IdentityError as exc:
        return _err("student_not_owned", detail=exc.message, status_code=exc.status_code)
    except Exception as exc:  # noqa: BLE001
        return _err("identity_check_failed", detail=str(exc))

    try:
        teacher_id = int(teacher_id)
    except (TypeError, ValueError):
        return _err("invalid_teacher_id", detail=f"teacher_id 必须是整数,收到 {teacher_id!r}")

    time_dt, error = _parse_scheduled_at(scheduled_at)
    if error:
        return _err("invalid_scheduled_at", detail=error)

    try:
        success, result = services.lesson.create_lesson(
            teacher_id=teacher_id,
            student_id=numeric_id,
            scheduled_at=time_dt,
            lesson_type=TRIAL_LESSON_TYPE,
            subject=subject,
            duration_minutes=TRIAL_DURATION_MINUTES,
            status=TRIAL_LESSON_STATUS,
            price=TRIAL_PRICE,
        )
    except Exception as exc:  # noqa: BLE001
        return _err("lesson_create_failed", detail=str(exc))
    if not success:
        return _err("lesson_create_rejected", detail=str(result))
    return _ok(lesson_id=result, teacher_id=teacher_id, student_id=numeric_id,
               scheduled_at=time_dt.isoformat(sep=" "),
               duration_minutes=TRIAL_DURATION_MINUTES)


# ----------------------------------------------------------------------
# MCP server 注册
# ----------------------------------------------------------------------


def _memoize(factory: Callable[[], Services]) -> Callable[[], Services]:
    """把零参工厂变成惰性单例:首次调用时构造,之后复用同一实例。"""
    box: Dict[str, Services] = {}

    def get() -> Services:
        if "svc" not in box:
            box["svc"] = factory()
        return box["svc"]

    return get


def create_server(
    db_path: Optional[str] = None,
    service_factory: Callable[[], Services] = build_services,
) -> MCPServer:
    """注册 5 个只读工具 + 1 个写入工具,返回 MCPServer(默认 stdio)。

    service_factory: 服务构造钩子(测试注入替身用);默认按 db_path / settings 构造。
    service 实例在首次工具调用时惰性创建并被本 server 复用。
    """
    mcp = MCPServer(name=MCP_SERVER_NAME, version=MCP_SERVER_VERSION)
    # service 惰性构造一次并被所有工具复用:构造会连库,不在 import/注册期做 IO
    make_services = service_factory if db_path is None else partial(service_factory, db_path)
    services: Callable[[], Services] = _memoize(make_services)

    @mcp.tool(name="find_teachers", title="匹配老师")
    def _find_teachers(subject: str, grade_level: str | None = None) -> str:
        """按学科匹配家教老师。subject: 学科(如"数学");grade_level: 年级段(如"高中"),不传则不限年级。"""
        return find_teachers(services(), subject, grade_level)

    @mcp.tool(name="get_teacher_availability", title="老师可约档期")
    def _get_teacher_availability(teacher_id: int) -> str:
        """查询某位老师的可约档期。teacher_id: find_teachers 返回的老师数字 ID。"""
        return get_teacher_availability(services(), teacher_id)

    @mcp.tool(name="search_knowledge", title="知识库检索")
    async def _search_knowledge(query: str, top_k: int = 3) -> str:
        """检索家教知识库(课程介绍/教学方法/政策等)。query: 检索文本;top_k: 返回条数,默认 3。"""
        return await search_knowledge(services(), query, top_k)

    @mcp.tool(name="list_packages", title="在售课程套餐")
    def _list_packages() -> str:
        """列出在售课程套餐(名称/学科/年级/课时数/价格),用于向家长介绍课程。"""
        return list_packages(services())

    @mcp.tool(name="get_student_lessons", title="学员即将到来的课时")
    def _get_student_lessons(student_id: str, limit: int = 5) -> str:
        """读取某学员即将到来的已确认课时。student_id: 学员 ID(字符串);limit: 返回条数,默认 5。需先通过 MCP_CALLER_USER_ID 声明身份。"""
        return get_student_lessons(services(), student_id, limit)

    @mcp.tool(name="book_trial_lesson", title="预约免费试讲")
    def _book_trial_lesson(
        teacher_id: int,
        student_id: str,
        scheduled_at: str,
        subject: str | None = None,
        write_token: str = "",
    ) -> str:
        """预约一次免费试讲课(默认关闭)。teacher_id: 老师 ID;student_id: 学员 ID(须归属调用方身份);scheduled_at: 开始时间 YYYY-MM-DD HH:MM;subject: 学科(可空);write_token: 写入令牌。"""
        return book_trial_lesson(
            services(), teacher_id, student_id, scheduled_at, subject, write_token
        )

    return mcp


def run_stdio() -> None:
    """以 stdio 传输启动 server(外部 Agent 以子进程方式接入)。"""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    run_stdio()
