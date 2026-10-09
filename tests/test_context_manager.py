"""
统一上下文管理(ContextManager + 5 类上下文)单元测试

覆盖 agents/context/*:
- 5 个上下文各自 build 输出正确(数据源全部 mock,不触网、不碰 DB)
- ContextManager 按 priority 排序、estimate_tokens / usage_report 可用
- MemoryContext 无数据时优雅降级为占位文案

本仓库未装 pytest-asyncio;异步用例统一走 _run(asyncio.run) 包装。
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from agents.context.base import BaseContext
from agents.context.manager import ContextManager
from agents.context.contexts.system_prompt import SystemPromptContext
from agents.context.contexts.conversation import ConversationContext
from agents.context.contexts.memory import MemoryContext
from agents.context.contexts.structured_output import StructuredOutputContext
from agents.context.contexts.tools import ToolContext


def _run(coro):
    """同步包装异步 coro(与 tests/test_reschedule_routing.py 约定一致)。"""
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# 1. system_prompt 上下文
# ----------------------------------------------------------------------

class TestSystemPromptContext:

    def test_build_contains_tutor_role(self):
        text = SystemPromptContext().build(manager=None)
        assert text
        assert "家教" in text or "K12" in text

    def test_name_and_priority(self):
        ctx = SystemPromptContext()
        assert ctx.name == "system_prompt"
        assert ctx.priority == 0


# ----------------------------------------------------------------------
# 2. conversation 上下文
# ----------------------------------------------------------------------

class TestConversationContext:

    def test_build_with_session_and_messages_formats_history(self):
        ctx = ConversationContext()
        session_id = "sess-1"
        history = ctx.get_history(session_id)
        history.add_message(HumanMessage(content="你好"))
        history.add_message(AIMessage(content="您好！"))

        text = ctx.build(manager=_Manager(session_id=session_id))

        assert "用户: 你好" in text
        assert "顾问: 您好！" in text

    def test_build_without_session_returns_placeholder(self):
        text = ConversationContext().build(manager=_Manager(session_id=None))
        assert "暂无历史" in text

    def test_build_with_empty_session_returns_placeholder(self):
        ctx = ConversationContext()
        text = ctx.build(manager=_Manager(session_id="empty-sess"))
        assert "暂无历史" in text


# ----------------------------------------------------------------------
# 3. memory 上下文
# ----------------------------------------------------------------------

class TestMemoryContext:

    def _mock_analyzer(self, prefs):
        analyzer = MagicMock()
        analyzer.analyze_user_preferences.return_value = prefs
        return analyzer

    def test_build_with_profile_renders_summary(self):
        analyzer = self._mock_analyzer({
            "favorite_teacher_id": 7,
            "favorite_subject": "数学",
            "favorite_duration": 60,
            "total_lessons": 5,
            "last_lesson_date": "2026-09-01",
        })
        text = MemoryContext(analyzer=analyzer).build(manager=None)

        assert "偏好老师ID: 7" in text
        assert "偏好学科: 数学" in text
        assert "60" in text
        assert "5 节" in text
        assert "距上次上课" in text

    def test_build_without_data_degrades_gracefully(self):
        analyzer = self._mock_analyzer(None)  # 无画像 → 降级
        text = MemoryContext(analyzer=analyzer).build(manager=None)
        assert text
        assert "暂无用户画像" in text

    def test_build_on_analyzer_exception_degrades_gracefully(self):
        analyzer = MagicMock()
        analyzer.analyze_user_preferences.side_effect = RuntimeError("db down")
        text = MemoryContext(analyzer=analyzer).build(manager=None)
        assert "暂无用户画像" in text


# ----------------------------------------------------------------------
# 4. structured_output 上下文
# ----------------------------------------------------------------------

class TestStructuredOutputContext:

    def test_build_contains_json_constraint(self):
        text = StructuredOutputContext().build(manager=None)
        assert text
        assert "纯 JSON" in text
        assert "snake_case" in text

    def test_name_and_priority(self):
        ctx = StructuredOutputContext()
        assert ctx.name == "structured_output"
        assert ctx.priority == 40


# ----------------------------------------------------------------------
# 5. tools 上下文
# ----------------------------------------------------------------------

class TestToolContext:

    def test_build_lists_service_capabilities(self):
        text = ToolContext().build(manager=None)
        assert "可用能力" in text
        assert "knowledge.search" in text
        assert "recommendation.recommend_teachers" in text
        assert "lesson.create_lesson" in text

    def test_composite(self):
        from agents.context.contexts.tools import SERVICE_CAPABILITIES
        assert not SERVICE_CAPABILITIES or all(
            name and desc for name, desc in SERVICE_CAPABILITIES
        )


# ----------------------------------------------------------------------
# 6. ContextManager:priority / estimate_tokens / usage_report
# ----------------------------------------------------------------------

class _Manager:
    """给上下文用的最小 manager 替身,只暴露 session_id。"""

    def __init__(self, session_id=None):
        self.session_id = session_id


class _FakeContext(BaseContext):
    """自定义上下文,便于验证优先级排序与空文本跳过。"""

    def __init__(self, name, priority, text="fake"):
        self.name = name
        self.priority = priority
        self._text = text

    def build(self, manager) -> str:
        return self._text


class _EmptyContext(BaseContext):
    """build 返回空串,应被 ContextManager 跳过。"""

    name = "empty"
    priority = 15

    def build(self, manager) -> str:
        return ""


class _ExplodingContext(BaseContext):
    """build 抛异常的上下文,不应拖垮整次组装。"""

    name = "exploding"
    priority = 5

    def build(self, manager) -> str:
        raise RuntimeError("boom")


class TestContextManager:

    def test_contexts_sorted_by_priority(self):
        mgr = ContextManager([
            _FakeContext("b", priority=20),
            _FakeContext("a", priority=0),
        ])
        assert [c.name for c in mgr.contexts] == ["a", "b"]

    def test_add_re_sorts(self):
        mgr = ContextManager([_FakeContext("a", priority=10)])
        mgr.add(_FakeContext("c", priority=0))
        assert [c.name for c in mgr.contexts] == ["c", "a"]

    def test_build_assembles_all_plus_task(self):
        mgr = ContextManager([
            _FakeContext("sys", priority=0, text="系统"),
            _FakeContext("tools", priority=30, text="能力"),
        ])
        output = mgr.build(task="数学怎么学？", session_id="s1")
        # 系统上下文在前,工具在后
        assert output.index("系统") < output.index("能力")
        assert "用户问题：数学怎么学？" in output

    def test_build_skips_empty_contexts(self):
        mgr = ContextManager([
            _FakeContext("sys", priority=0, text="系统"),
            _EmptyContext(),
        ])
        output = mgr.build(task="问题")
        assert "系统" in output
        assert "empty" not in output

    def test_build_tolerates_exploding_context(self):
        mgr = ContextManager([
            _FakeContext("sys", priority=0, text="系统"),
            _ExplodingContext(),
        ])
        output = mgr.build(task="问题")
        assert "系统" in output  # 异常上下文被跳过,组装仍成功

    def test_estimate_tokens_is_positive_for_nonempty(self):
        mgr = ContextManager([_FakeContext("sys", priority=0, text="能力描述" * 10)])
        mgr.build(task="问题")
        assert mgr.estimate_tokens() > 0

    def test_usage_report_contains_name_chars_priority_preview(self):
        mgr = ContextManager([
            _FakeContext("sys", priority=0, text="能力描述"),
        ])
        mgr.build()
        report = mgr.usage_report()
        assert "sys" in report
        info = report["sys"]
        assert set(info) == {"chars", "priority", "preview"}
        assert info["priority"] == 0
        assert info["chars"] == len("能力描述")
        assert info["preview"] == "能力描述"


# ----------------------------------------------------------------------
# 7. 端到端:组装一个真实 5 类上下文(记忆数据源 mock),产出完整输入
# ----------------------------------------------------------------------

class TestContextManagerIntegration:

    def test_full_pipeline_build(self):
        analyzer = MagicMock()
        analyzer.analyze_user_preferences.return_value = {
            "favorite_teacher_id": 3,
            "favorite_subject": "英语",
            "favorite_duration": 45,
            "total_lessons": 8,
            "last_lesson_date": "2026-09-05",
        }
        mgr = ContextManager([
            SystemPromptContext(),
            ConversationContext(),
            MemoryContext(analyzer=analyzer),
            ToolContext(),
            StructuredOutputContext(),
        ])
        mgr.session_id = "s-full"

        output = mgr.build(task="给孩子补习英语", extra_contexts={"knowledge": "RAG片段"})

        assert "家教" in output                      # system_prompt
        assert "英语" in output                       # memory
        assert "knowledge.search" in output           # tools
        assert "纯 JSON" in output                     # structured_output
        assert "RAG片段" in output                     # 额外知识上下文
        assert "用户问题：给孩子补习英语" in output     # task

        # 已组装过一次,可估算 token
        assert mgr.estimate_tokens() > 0
        # 报告里 5 类上下文都在
        report = mgr.usage_report()
        assert {"system_prompt", "conversation", "memory", "tools", "structured_output"} <= set(report)


# ----------------------------------------------------------------------
# 8. Token 压缩钩子（P1）
# ----------------------------------------------------------------------

class TestTokenCompression:

    def _build_manager_with_history(self, rounds=20, session_id="sess-compress"):
        """构造含长会话历史的 manager(system + conversation)。"""
        mgr = ContextManager([
            _FakeContext("sys", priority=0, text="系统提示词" * 20),
            ConversationContext(),
        ])
        mgr.session_id = session_id
        conv = mgr.contexts[1]
        history = conv.get_history(session_id)
        for i in range(rounds):
            history.add_message(HumanMessage(content=f"问题{i}"))
            history.add_message(AIMessage(content=f"回答{i}"))
        return mgr

    def test_no_threshold_builds_full_history(self):
        mgr = self._build_manager_with_history()
        out = mgr.build(task="当前问题")
        assert "问题19" in out           # 最近内容在
        assert "问题0" in out            # 最旧内容也在(未压缩)
        assert "[更早的对话已折叠]" not in out

    def test_over_threshold_triggers_compress(self):
        mgr = self._build_manager_with_history()
        from agents.context.types import ContextBudgetExceeded
        # 必需系统本身超过10,不能删系统/当前问题来假装压缩成功。
        with pytest.raises(ContextBudgetExceeded):
            mgr.build(task="当前问题", max_tokens=10)
        out = mgr.build(task="当前问题", max_tokens=800)
        assert mgr.estimate_tokens() <= 800
        assert "[更早的对话已折叠]" in out
        assert "问题19" in out
        assert "用户问题：当前问题" in out

    def test_compress_reduces_output_size(self):
        mgr = self._build_manager_with_history()
        full = mgr.build(task="当前问题")
        compressed = mgr.build(task="当前问题", max_tokens=800)
        assert mgr.estimate_tokens() <= 800
        assert "系统提示词" * 20 in compressed
        assert "用户问题：当前问题" in compressed
        assert len(compressed) < len(full)