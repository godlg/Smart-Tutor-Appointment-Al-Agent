"""
统一上下文管理包

实现 5 类上下文(system_prompt / conversation / memory / tools / structured_output)
的构建与统一组装,收敛散落在各 Agent 的上下文管理逻辑。
本包属于 Agent 层,可依赖 services/;不允许被 services/ 反向依赖。

P0 试点:仅 ConsultantAgent 接入,其它 Agent 本次不动。
"""

from .base import BaseContext
from .manager import ContextManager
from .types import (
    ContextBuildResult,
    ContextBudget,
    ContextBudgetExceeded,
    estimate_token_count,
    count_tokens,
    PRIORITY_SYSTEM_PROMPT,
    PRIORITY_CONVERSATION,
    PRIORITY_MEMORY,
    PRIORITY_TOOLS,
    PRIORITY_STRUCTURED_OUTPUT,
    DEFAULT_USER_ID,
)
from .contexts.system_prompt import SystemPromptContext, SYSTEM_PROMPT_TEXT
from .contexts.conversation import ConversationContext
from .contexts.memory import MemoryContext
from .contexts.structured_output import StructuredOutputContext
from .contexts.tools import ToolContext

__all__ = [
    "BaseContext",
    "ContextManager",
    "ContextBuildResult",
    "ContextBudget",
    "ContextBudgetExceeded",
    "estimate_token_count",
    "count_tokens",
    "PRIORITY_SYSTEM_PROMPT",
    "PRIORITY_CONVERSATION",
    "PRIORITY_MEMORY",
    "PRIORITY_TOOLS",
    "PRIORITY_STRUCTURED_OUTPUT",
    "DEFAULT_USER_ID",
    "SystemPromptContext",
    "SYSTEM_PROMPT_TEXT",
    "ConversationContext",
    "MemoryContext",
    "StructuredOutputContext",
    "ToolContext",
]