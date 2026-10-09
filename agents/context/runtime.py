"""正常聊天使用的上下文装配;各Agent共读唯一历史,不拥有可见消息写权限。"""
from config.identity import Identity
from config.memory import (
    CONTEXT_MAX_TOKENS, CONTEXT_OUTPUT_TOKENS, CONTEXT_KEEP_RECENT_ROUNDS,
    CONTEXT_ROLLUP_ROUNDS, CONTEXT_SUMMARY_MAX_CHARS,
)
from .manager import ContextManager
from .contexts.system_prompt import SystemPromptContext
from .contexts.conversation import ConversationContext
from .contexts.memory import MemoryContext
from .contexts.structured_output import StructuredOutputContext
from .contexts.tools import ToolContext


def build_runtime_context(identity: Identity, history, conversation_service,
                          include_memory: bool = False) -> ContextManager:
    contexts = [
        SystemPromptContext(),
        ConversationContext(keep_rounds=CONTEXT_KEEP_RECENT_ROUNDS,
                            history_provider=lambda sid: conversation_service.get_history(sid)),
        StructuredOutputContext(), ToolContext(),
    ]
    if include_memory:
        contexts = contexts + [MemoryContext(identity=identity,
                                             memory_service=conversation_service.memory_service)]
    manager = ContextManager(
        contexts, identity=identity, conversation_service=conversation_service,
        rollup_enabled=True, rollup_rounds=CONTEXT_ROLLUP_ROUNDS,
        keep_recent_rounds=CONTEXT_KEEP_RECENT_ROUNDS,
        output_tokens=CONTEXT_OUTPUT_TOKENS, summary_max_chars=CONTEXT_SUMMARY_MAX_CHARS,
    )
    manager.session_id = identity.session_id
    manager.max_tokens = CONTEXT_MAX_TOKENS
    return manager
