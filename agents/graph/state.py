"""LangGraph 主控编排状态定义（P0-A）。

只新增/增补，不改任何现有编排层。字段刻意与现有 flow 概念对齐：
- messages: LangGraph add_messages 归约，承载人机对话流
- flow_state: 复用 config.constants.StateEnum 的 .value
- intent: 10 类意图之一（与 TaskClassifier 归一结果一致）
- slots: 业务槽位（subject / grade_level / start_time ...），可留空
- session_id: 会话隔离键，同时用作 checkpointer 的 thread_id
- handoff: 是否已进入人工接管
- error: 本轮是否触发异常兜底（等价 legacy 最外层 except，命中即终止本轮，不再路由下游）
- verification_required / busy / unrelated_count: 横切判定结果（P0-A 补齐）
- reply_flags: 需要前置的标记（如 [HANDOFF]），保持流式协议原样外发
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from config.constants import StateEnum


class TutorState(TypedDict):
    """主控编排状态（跨节点共享，由 checkpointer 持久化）。"""

    messages: Annotated[list, add_messages]
    flow_state: str
    intent: str
    slots: dict
    session_id: str
    handoff: bool
    error: bool
    verification_required: bool
    busy: bool
    unrelated_count: int
    reply_flags: list


def new_state(user_input: str, session_id: str, slots: Optional[dict] = None,
              flow_state: Optional[str] = None) -> TutorState:
    """构造一次图调用的初始状态（便捷工厂）。

    flow_state: 每轮请求前从图 state 恢复的续轮态（InMemorySaver 同进程内有效）。
    不传时按新会话处理，落到 CLASSIFY 走分类。
    """
    return {
        "messages": [{"role": "human", "content": user_input}],
        "flow_state": flow_state or StateEnum.CLASSIFY.value,
        "intent": "",
        "slots": dict(slots or {}),
        "session_id": session_id,
        "handoff": False,
        "error": False,
        "verification_required": False,
        "busy": False,
        "unrelated_count": 0,
        "reply_flags": [],
    }
