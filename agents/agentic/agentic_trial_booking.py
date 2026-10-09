"""Agentic 试讲预约工具循环（面试/演示用独立能力）。

与现有『LLM 分类 + 规则状态机』主路径完全解耦：
- 这是新增的**工具调用闭环**（LLM 自主决策 → 调用 tool → 真实落库），
  不改动 agents/task_classification / agents/trial 等 434 测试路径。
- `@tool book_trial_lesson` 完成「冲突预检 + 真实落库」；
  `run_agentic_trial_booking` 跑完整多轮工具循环（LLM 决策→执行→ToolMessage→最终自然语言回复）。

面试可讲点：
1. 工具 description / 参数 schema 质量——直接影响 LLM 调用正确率；
2. 完整 tool loop（多轮：第一次 invoke 拿到 tool_calls → 执行 → ToolMessage 喂回 → 再 invoke）；
3. 失败降级齐全——LLM 不调工具 / 缺参 / 时间冲突 / 落库失败 各自有明确分支。

服务层支持注入（测试 / 8003 演示用临时 DB），不污染 data/smart_appointment.db。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from observability import trace_span


def _coerce_naive(dt: datetime) -> datetime:
    """统一 strip tzinfo，与 lesson_repo 内部 naive 比较一致（同 _create_pending_trial_lesson）。"""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _default_lesson_service():
    from services.lesson_service import LessonService

    return LessonService()


def _default_student_service():
    from services.student_profile_service import StudentProfileService

    return StudentProfileService()


def _make_book_trial_lesson_tool(lesson_service=None, student_service=None):
    """构造 `@tool book_trial_lesson`——绑定可注入的服务（默认真实服务）。

    用闭包绑定便于测试 / 8003 演示注入临时 DB 的 service，
    不需要改任何现有模块。
    """
    ls = lesson_service or _default_lesson_service()
    ss = student_service or _default_student_service()

    @tool
    def book_trial_lesson(
        teacher_id: int,
        time_iso: str,
        subject: Optional[str] = None,
        duration_minutes: int = 30,
    ) -> str:
        """为家长预约一次免费的试讲课，并尽力真实落库。

        参数：
          - teacher_id: 要预约老师的数字 ID（上下游已确认好的老师）。
          - time_iso: 试讲开始时间，格式 YYYY-MM-DD HH:MM，如 "2026-09-20 19:00"
            （也兼容 ISO 8601 "2026-09-20T19:00:00"），按北京时间解释。
          - subject: 学科（可空，如 "数学"）。
          - duration_minutes: 试讲时长（分钟），默认 30。

        返回：
          - 成功："OK: lesson_id=<数字>"。
          - 失败："ERROR: <原因>"，例如老师该时段已有课程、时间无法解析、默认学员创建失败。
        注意：调用前**必先检查该老师该时段是否已有课程**，有冲突时返回 ERROR，
        以便继续引导家长更换时间。
        """
        try:
            raw = str(time_iso).strip()
            time_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return f"ERROR: 无法解析预约时间 {time_iso!r}，请使用 YYYY-MM-DD HH:MM 格式"
        # 与 _create_pending_trial_lesson / check_teacher_conflict 一致：strip tzinfo
        time_dt = _coerce_naive(time_dt)
        end_time = time_dt + timedelta(minutes=duration_minutes)

        # 0) 冲突预检：老师该时段已有课程 → 返回 ERROR，让 LLM 生成换时间引导
        if ls.lesson_repo.check_teacher_conflict(teacher_id, time_dt, end_time):
            return "ERROR: 老师在该时段已有课程，请换时间"

        # 1) 默认学员兜底（试讲流程里没有显式选学员）
        student = ss.initialize_default_student()
        if not student or "id" not in student:
            return "ERROR: 默认学员创建失败"

        # 2) 真实落库（type=trial，直接 confirmed，免费）
        success, result = ls.create_lesson(
            teacher_id=teacher_id,
            student_id=student["id"],
            scheduled_at=time_dt,
            lesson_type="trial",
            subject=subject,
            duration_minutes=duration_minutes,
            status="confirmed",
            price=0.0,
        )
        if not success:
            return f"ERROR: {result}"
        return f"OK: lesson_id={result}"

    return book_trial_lesson


def run_agentic_trial_booking(
    llm,
    teacher_id: Optional[int] = None,
    time_iso: Optional[str] = None,
    subject: Optional[str] = None,
    lesson_service=None,
    student_service=None,
    user_message: str = "请帮我预约试讲",
) -> Dict[str, Any]:
    """完整工具调用循环：LLM 自主决策 → 执行 tool → ToolMessage 喂回 → 最终自然语言回复。

    Args:
        llm: 支持 bind_tools 的 chat model（真实用 config.model_provider.create_chat_model）。
        teacher_id / time_iso / subject: 已在 API 层解析好的预约上下文，
              注入系统提示，让 LLM 有足够信息发起 tool 调用（绝对时间场景 LLM 会自主返回 tool_calls）。
        lesson_service / student_service: 可注入（临时 DB），None 时用真实服务。
        user_message: 家长本轮原始输入（作为 HumanMessage）。

    Returns:
        {"tool_called": bool, "lesson_id": int|None, "message": str}
        - message 永远是给家长的自然语言最终回复（工具结果已由 LLM 转述）。
        - tool_called=False 表示 LLM 本轮选择了直接对话 / 缺参未调工具（调用方决定回退）。
    """
    book_trial = _make_book_trial_lesson_tool(lesson_service, student_service)
    llm_with_tools = llm.bind_tools([book_trial])

    system = (
        "你是家教平台「试讲预约」助手，负责帮家长预约免费试讲。\n"
        "当预约信息齐备时，调用 book_trial_lesson 工具完成预约，"
        "并基于工具返回结果用简体中文向家长礼貌地汇报。\n"
        "若工具返回 ERROR（例如老师该时段已有课程），请向家长解释并引导其更换一个时间。\n"
        f"当前待办预约：老师ID={teacher_id}，时间={time_iso}，学科={subject or '未知'}。"
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ]

    first = llm_with_tools.invoke(messages)
    tool_calls = getattr(first, "tool_calls", None) or []
    if not tool_calls:
        # 降级分支：LLM 选择不调工具（缺参 / 直接对话）→ 调用方决定回退
        return {
            "tool_called": False,
            "lesson_id": None,
            "message": getattr(first, "content", "") or "",
        }

    # 多轮 tool loop：把带 tool_calls 的 assistant 消息 + 各 ToolMessage 一并购入，再喂回
    messages.append(AIMessage(content=first.content, tool_calls=tool_calls))
    lesson_id: Optional[int] = None
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args") or {}
        # Trace 埋点:工具调用(参数摘要,避免敏感/超长内容进日志)
        arg_summary = {
            k: (str(v)[:20] + "..." if len(str(v)) > 20 else v)
            for k, v in args.items()
        }
        with trace_span("tool_call", tool=name, args=arg_summary) as span:
            if name == "book_trial_lesson":
                out = book_trial.invoke(args)
            else:
                out = f"ERROR: 未知工具 {name}"
            ok = isinstance(out, str) and out.startswith("OK:")
            span.attrs["ok"] = ok
            span.attrs["result"] = out[:50] if isinstance(out, str) else str(out)[:50]
            if not ok:
                span.attrs["error"] = True
        if isinstance(out, str) and out.startswith("OK: lesson_id="):
            try:
                lesson_id = int(out.split("=", 1)[1])
            except ValueError:
                lesson_id = None
        messages.append(ToolMessage(content=str(out), tool_call_id=tc.get("id")))

    final = llm_with_tools.invoke(messages)
    return {
        "tool_called": True,
        "lesson_id": lesson_id,
        "message": getattr(final, "content", "") or "",
    }