"""需求草稿检查点：恢复信息，不恢复执行权或旧确认。

仅依赖 config 常量与 duck-typed 状态接口，不导入 agents 或业务服务。
身份/会话 ownership 校验、恢复后的输入拦截由主控负责；本模块不执行业务操作。
时间字段只供展示，恢复时仍需重新询问时间、查询老师/档期并确认。
"""

import json
import logging
from copy import deepcopy
from typing import Any

from config.constants import StateEnum

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
MAX_JSON_CHARS = 16384
# 长度以 Python Unicode 字符数计算；超限拒绝，不截断姓名或需求。
SAFE_FIELD_LIMITS = {
    "subject": 40,
    "grade_level": 40,
    "preferred_style": 40,
    "preferred_teacher_name": 80,
    "student_situation": 500,
}
DISPLAY_FIELD_LIMITS = {"start_time": 80, "duration_minutes": 20}
DRAFT_KINDS = ("trial", "lesson", "recommendation")
BUSINESS_FLOWS = {
    StateEnum.TRIAL.value,
    StateEnum.LESSON.value,
    StateEnum.RECOMMENDATION_CONFIRM.value,
    "trial_confirm",
    "trial_convert",
}
RECONFIRMATION_NOTICE = (
    "已恢复之前的需求草稿，但旧的预约、支付、调课或退课指令已清空，不会自动执行。"
    "草稿中的老师和时间仅供参考，请重新说明本次需求、老师及时间，重新查询并重新确认。"
)
INVALID_NOTICE = "旧检查点无法安全恢复，待执行状态已清空。请重新说明需求和时间并重新确认。"


def _empty_drafts() -> dict:
    return {kind: {} for kind in DRAFT_KINDS}


def _flow_value(value: Any) -> str:
    value = getattr(value, "value", value)
    return value if isinstance(value, str) and len(value) <= 64 else "unknown"


def _source_flow(agent: Any) -> str:
    state = getattr(agent, "state", None)
    if state is None:
        manager = getattr(agent, "state_manager", None)
        state = getattr(manager, "state", None)
    return _flow_value(getattr(state, "value", state))


def _validated_fields(history: Any) -> dict:
    if not isinstance(history, dict):
        return {}
    allowed = {**SAFE_FIELD_LIMITS, **DISPLAY_FIELD_LIMITS}
    draft = {}
    for field, limit in allowed.items():
        value = history.get(field)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or len(value) > limit or value.lower() in {"未知", "unknown", "null", "none"}:
            continue
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            continue
        draft[field] = value
    return draft


def _recommendation_sid(agent: Any, child: Any) -> str:
    return getattr(child, "session_id", None) or getattr(agent, "session_id", "default")


def _work_histories(agent: Any) -> dict:
    result = {}
    for kind in ("trial", "lesson"):
        child = getattr(agent, f"{kind}_agent", None)
        result[kind] = getattr(child, f"{kind}_history", {})
    child = getattr(agent, "recommendation_agent", None)
    histories = getattr(child, "history", {})
    result["recommendation"] = (
        histories.get(_recommendation_sid(agent, child), {})
        if isinstance(histories, dict) else {}
    )
    return result


def _has_operation(history: Any) -> bool:
    if not isinstance(history, dict):
        return False
    return any(
        bool(value) and isinstance(key, str) and (
            key.startswith(("pending", "awaiting", "confirmed", "convert", "package"))
            or key in {"teacher_id", "selected_package", "last_recommendations"}
        )
        for key, value in history.items()
    )


def _has_requirements(drafts: dict) -> bool:
    # reset 自带的 30/60 分钟默认时长不是待执行需求。
    return any(any(key != "duration_minutes" for key in draft) for draft in drafts.values())


def capture_checkpoint(agent: Any) -> dict:
    """捕获 v1、JSON 可序列化且与原状态分离的白名单展示草稿。"""
    histories = _work_histories(agent)
    drafts = {kind: _validated_fields(histories[kind]) for kind in DRAFT_KINDS}
    source_flow = _source_flow(agent)
    return {
        "version": CHECKPOINT_VERSION,
        "source_flow": source_flow,
        "drafts": drafts,
        "requires_reconfirmation": (
            source_flow in BUSINESS_FLOWS or _has_requirements(drafts)
            or any(_has_operation(history) for history in histories.values())
        ),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_JSON_CHARS:
            raise ValueError("检查点超过安全大小上限")
        return json.loads(value)
    return value


def _decode_checkpoint(checkpoint: Any) -> tuple[dict | None, bool]:
    """一次验证完整结构后才允许回填；pending_ops 只识别有无，绝不重放。"""
    if not isinstance(checkpoint, dict):
        return None, False
    try:
        payload = _json_value(checkpoint.get("payload"))
        pending_ops = _json_value(checkpoint.get("pending_ops"))
    except (TypeError, ValueError, RecursionError):
        return None, False
    if pending_ops is not None and not isinstance(pending_ops, list):
        return None, False
    if not isinstance(payload, dict) or type(payload.get("version")) is not int:
        return None, bool(pending_ops)
    if payload["version"] != CHECKPOINT_VERSION:
        return None, bool(pending_ops)
    drafts = payload.get("drafts")
    if not isinstance(drafts, dict):
        return None, bool(pending_ops)
    if any(not isinstance(drafts.get(kind, {}), dict) for kind in DRAFT_KINDS):
        return None, bool(pending_ops)
    return payload, bool(pending_ops)


def _booking_defaults(kind: str) -> dict:
    defaults = {
        "subject": None, "grade_level": None, "start_time": None,
        "duration_minutes": "30" if kind == "trial" else "60",
        "preferred_teacher_name": None, "preferred_style": None,
        "unrelated_escalations": 0,
    }
    return {**defaults, **({"student_situation": None} if kind == "trial" else {"package_id": None})}


def _detach_shared_history(agent: Any, child: Any) -> None:
    """reset 只能清局部提取历史，即使误将共享历史/其 messages 作为局部别名。"""
    local = getattr(child, "chat_history", None)
    if local is None:
        return
    shared_histories = (
        getattr(agent, "session_chat_history", None),
        getattr(child, "session_chat_history", None),
    )
    local_messages = local if isinstance(local, list) else getattr(local, "messages", None)
    is_shared = any(
        shared is not None and (
            local is shared or (
                local_messages is not None
                and local_messages is (
                    shared if isinstance(shared, list) else getattr(shared, "messages", None)
                )
            )
        ) for shared in shared_histories
    )
    if not is_shared:
        return
    child.chat_history = deepcopy(local)
    histories = getattr(child, "chats_by_session_id", None)
    if isinstance(histories, dict):
        child.chats_by_session_id = {
            sid: child.chat_history if history is local else history
            for sid, history in histories.items()
        }


def _clear_booking_state(agent: Any, child: Any, kind: str) -> None:
    # 先替换执行状态，局部上下文清理失败也不能遗留旧 pending。
    setattr(child, f"{kind}_history", _booking_defaults(kind))
    child.finished = False
    _detach_shared_history(agent, child)
    local = getattr(child, "chat_history", None)
    if local is not None:
        local.clear()


def _reset_booking_state(agent: Any, child: Any, kind: str) -> bool:
    try:
        _detach_shared_history(agent, child)
        reset = getattr(child, "reset", None)
        if callable(reset):
            reset()
        else:
            _clear_booking_state(agent, child, kind)
        return True
    except Exception as exc:
        logger.warning("checkpoint %s reset failed (%s); restore invalidated",
                       kind, type(exc).__name__)
        try:
            _clear_booking_state(agent, child, kind)
        except Exception as cleanup_error:
            logger.warning("checkpoint %s local history cleanup failed (%s)",
                           kind, type(cleanup_error).__name__)
        return False


def _reset_recommendation_state(agent: Any, child: Any) -> bool:
    is_reset = True
    defaults = {
        "last_recommendations": [], "subject": None, "grade_level": None,
        "preferred_style": None, "awaiting_confirmation": False,
        "confirmed_teacher": None, "recommendation_declined": False,
    }
    try:
        factory = getattr(child, "_default_history", None)
        if callable(factory):
            defaults = factory()
    except Exception as exc:
        logger.warning("checkpoint recommendation reset failed (%s); restore invalidated",
                       type(exc).__name__)
        is_reset = False
    histories = getattr(child, "history", {})
    child.history = {
        **(histories if isinstance(histories, dict) else {}),
        _recommendation_sid(agent, child): defaults,
    }
    return is_reset


def _reset_work_state(agent: Any) -> bool:
    for state in (
        getattr(agent, "state", None),
        getattr(getattr(agent, "state_manager", None), "state", None),
    ):
        if state is not None and hasattr(state, "value"):
            state.value = StateEnum.CLASSIFY
    is_reset = True
    for kind in ("trial", "lesson"):
        child = getattr(agent, f"{kind}_agent", None)
        if child is not None:
            # 调用置于 and 左侧：前一个 child 失败也必须继续清理后续 child。
            is_reset = _reset_booking_state(agent, child, kind) and is_reset
    child = getattr(agent, "recommendation_agent", None)
    if child is not None:
        is_reset = _reset_recommendation_state(agent, child) and is_reset
    return is_reset


def _refill_requirements(agent: Any, drafts: dict) -> None:
    for kind in DRAFT_KINDS:
        child = getattr(agent, f"{kind}_agent", None)
        if child is None:
            continue
        safe = {key: value for key, value in drafts[kind].items() if key in SAFE_FIELD_LIMITS}
        if kind == "recommendation":
            sid = _recommendation_sid(agent, child)
            child.history = {**child.history, sid: {**child.history[sid], **safe}}
        else:
            attribute = f"{kind}_history"
            setattr(child, attribute, {**getattr(child, attribute), **safe})


def restore_checkpoint(agent: Any, checkpoint: Any) -> dict:
    """消费 load_checkpoint 的 {state,payload(JSON),pending_ops(JSON)} 包。

    无论版本/结构是否有效均先清除工作状态。仅回填安全字符串需求；时间与
    时长保留在返回 drafts 中，不回填业务 history，调用方必须拦截旧确认输入。
    任一 child 清理失败使整体恢复失效，不回填需求草稿。
    """
    current_requires = capture_checkpoint(agent)["requires_reconfirmation"]
    payload, has_pending = _decode_checkpoint(checkpoint)
    is_reset = _reset_work_state(agent)
    if payload is None or not is_reset:
        requires = checkpoint is not None or current_requires or not is_reset
        return {"notice": INVALID_NOTICE if requires else "",
                "requires_reconfirmation": requires, "drafts": _empty_drafts()}
    drafts = {kind: _validated_fields(payload["drafts"].get(kind, {})) for kind in DRAFT_KINDS}
    requires = (
        current_requires or payload.get("requires_reconfirmation") is True or has_pending
        or _flow_value(checkpoint.get("state")) in BUSINESS_FLOWS
        or _flow_value(payload.get("source_flow")) in BUSINESS_FLOWS
        or _has_requirements(drafts)
        or any(_has_operation(payload["drafts"].get(kind, {})) for kind in DRAFT_KINDS)
    )
    _refill_requirements(agent, drafts)
    return {"notice": RECONFIRMATION_NOTICE if requires else "",
            "requires_reconfirmation": requires, "drafts": drafts}
