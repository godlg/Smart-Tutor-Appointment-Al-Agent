"""
人工接管 API（客服侧）

- GET  /api/handoffs?status=pending            拉取转人工记录(默认 pending)
- POST /api/handoffs/{handoff_id}/resolve      标记某条记录已处理
- POST /api/handoffs/{handoff_id}/claim        客服认领工单
- POST /api/handoffs/{handoff_id}/reply        客服回复家长(不 resolve)
- GET  /api/handoffs/{handoff_id}/messages     工单全部对话(升序)
- GET  /api/handoffs/session/{session_id}/messages  家长侧拉取(升序,含客服回复)
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .core.response_models import DataResponse

router = APIRouter(prefix="/api/handoffs", tags=["人工接管"])


class ResolveHandoffRequest(BaseModel):
    """标记已处理请求体，字段全可选。"""
    handled_by: Optional[str] = None
    note: Optional[str] = None


class ClaimHandoffRequest(BaseModel):
    """认领请求体。"""
    agent_id: str


class ReplyHandoffRequest(BaseModel):
    """客服回复请求体。"""
    agent_id: str
    content: str


def _serialize(record: dict) -> dict:
    """把 datetime 字段转成 ISO 字符串，便于 JSON 序列化。"""
    out = dict(record)
    for key in ("created_at", "resolved_at"):
        value = out.get(key)
        if value is not None and hasattr(value, "isoformat"):
            out[key] = value.isoformat()
    return out


@router.get("", response_model=DataResponse, summary="拉取转人工记录")
async def list_handoffs(status: Optional[str] = "pending"):
    """按状态拉取转人工记录（默认只返回待处理的 pending）。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService()
    records = svc.get_all(status=status) if status else svc.get_all()
    return DataResponse(
        message="获取转人工记录成功",
        data=[_serialize(r) for r in records],
    )


@router.get("/session/{session_id}/messages", response_model=DataResponse, summary="家长侧拉取会话消息")
async def list_session_messages(session_id: str):
    """家长侧拉取该会话全部消息（升序，含客服回复）。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService()
    messages = svc.get_messages_by_session(session_id)
    return DataResponse(
        message="获取会话消息成功",
        data=[_serialize(m) for m in messages],
    )


@router.get("/{handoff_id}/messages", response_model=DataResponse, summary="工单全部对话")
async def list_handoff_messages(handoff_id: int):
    """拉取某工单的全部对话（升序）。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService()
    messages = svc.get_messages(handoff_id)
    return DataResponse(
        message="获取工单对话成功",
        data=[_serialize(m) for m in messages],
    )


@router.post("/{handoff_id}/claim", response_model=DataResponse, summary="客服认领工单")
async def claim_handoff(handoff_id: int, request: ClaimHandoffRequest):
    """客服认领工单（已认领时幂等返回原客服）。"""
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService()
    record = svc.claim(handoff_id, request.agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="转人工记录不存在")
    return DataResponse(message="认领成功", data=_serialize(record))


@router.post("/{handoff_id}/reply", response_model=DataResponse, summary="客服回复家长")
async def reply_handoff(handoff_id: int, request: ReplyHandoffRequest):
    """客服回复落库（不 resolve），并（图路径下）恢复挂起的图。

    - legacy：仅落库（`role='agent'`），家长靠轮询 `GET session/{id}/messages` 拿回复。
    - 图路径（`ORCHESTRATOR=graph`）且该 thread 仍在挂起态：落库后用
      `Command(resume=content)` 恢复图，把恢复产生的输出同样落库；返回值额外带
      `resumed` / `resume_output` 字段（legacy 下不出现，行为逐字不变）。
    """
    from services.human_handoff_service import HumanHandoffService

    svc = HumanHandoffService()
    message = svc.agent_reply(handoff_id, request.agent_id, request.content)
    if message is None:
        raise HTTPException(status_code=404, detail="转人工记录不存在或已处理")
    data = _serialize(message)

    from .chat_handler import resume_graph_handoff
    output = await resume_graph_handoff(message["session_id"], request.content)
    if output is not None:
        data["resumed"] = True
        data["resume_output"] = output
    return DataResponse(message="回复成功", data=data)


@router.post("/{handoff_id}/resolve", response_model=DataResponse, summary="标记转人工记录已处理")
async def resolve_handoff(handoff_id: int, request: ResolveHandoffRequest = None):
    """把某条转人工记录标记为已处理（status=resolved）。

    图路径下：若该 thread 仍挂起，用 `Command(resume=HANDOFF_RESOLVED)` 结束挂起，
    使其恢复"正常 AI 对话"（等价 legacy resolve 后解除内存短路的行为）；legacy 下不变。
    """
    from services.human_handoff_service import HumanHandoffService

    request = request or ResolveHandoffRequest()
    svc = HumanHandoffService()
    ok = svc.resolve(handoff_id, handled_by=request.handled_by, note=request.note)
    if not ok:
        raise HTTPException(status_code=404, detail="转人工记录不存在")
    record = svc.repo.get_by_id(handoff_id)
    data = _serialize(record)

    from agents.graph import HANDOFF_RESOLVED
    from .chat_handler import release_graph_handoff, resume_graph_handoff
    # 图路径：先解除会话级 handoff 内存短路（等价 legacy resolve 的 discard），
    # 再用哨兵结束挂起 thread，使家长下一条消息回到普通 AI 对话。
    release_graph_handoff(record["session_id"])
    output = await resume_graph_handoff(record["session_id"], HANDOFF_RESOLVED)
    if output is not None:
        data["resumed"] = True
        data["resume_output"] = output
    return DataResponse(
        message="转人工记录已处理",
        data=data,
    )
