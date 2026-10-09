"""
分层记忆增强单元测试

覆盖 agents/context + services/long_term_memory_service 的新增能力:
- 评分公式: score = 0.6*cos_sim + 0.3*time_decay + 0.1*importance (权重正确性)
- 时间衰减: exp(-days/30) 单调递减
- 召回 Top-5 截断 + 相似度下限过滤
- 记忆持久化: 新增 / 查询 / 更新 / 删除
- 60% 窗口触发滚动摘要、<60% 不触发
- 摘要写回长期记忆 + 不重复摘要
- 无记忆 / 无 embedding 时回退原画像摘要(兼容路径)

Embedding 用确定性 fake(sha256→16 维),不依赖远程 API。
"""
import asyncio
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def _fake_embed(text: str) -> List[float]:
    """sha256→16 维确定性向量,相似文本向量相近。"""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input(services 层),防止触网。"""
    from services import text_embedding
    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)


@pytest.fixture
def memory_service():
    """基于内存 SQLite 的长期记忆服务。"""
    from services.long_term_memory_service import LongTermMemoryService
    svc = LongTermMemoryService(db_path="sqlite:///:memory:")
    yield svc
    svc.db_router.close()


def _manager(session_id="sess-1", task=""):
    """最小 manager 替身,暴露 session_id 与 task。"""
    class _M:
        def __init__(self):
            self.session_id = session_id
            self.task = task
    return _M()


# ----------------------------------------------------------------------
# 1. 评分公式(0.6 / 0.3 / 0.1)
# ----------------------------------------------------------------------

class TestScoreFormula:

    def test_weights_apply_correctly(self):
        from services.long_term_memory_service import score_memory, _cos_sim
        q = _fake_embed("孩子想补数学")
        mem = {
            "content": "偏好数学",
            "embedding": q,  # cos_sim = 1.0
            "importance": 0.5,
            "created_at": datetime.utcnow(),  # decay ≈ 1.0
        }
        score = score_memory(
            q, mem,
            weight_sim=0.6, weight_recency=0.3, weight_importance=0.1,
        )
        # 0.6*1.0 + 0.3*~1.0 + 0.1*0.5 ≈ 0.95
        assert 0.6 * 1.0 + 0.3 * 0.99 + 0.1 * 0.5 - 0.01 <= score <= 0.6 * 1.0 + 0.3 * 1.0 + 0.1 * 0.5 + 0.01

    def test_weights_control_ordering(self):
        """同向量相似度下,重要度高者优先;同重要度下,相似度高者优先。"""
        from services.long_term_memory_service import score_memory
        q = _fake_embed("数学一对一")
        now = datetime.utcnow()

        high_importance = score_memory(
            q, {"embedding": q, "importance": 1.0, "created_at": now},
            weight_sim=0.6, weight_recency=0.3, weight_importance=0.1,
        )
        low_importance = score_memory(
            q, {"embedding": q, "importance": 0.0, "created_at": now},
            weight_sim=0.6, weight_recency=0.3, weight_importance=0.1,
        )
        # 重要度差 1.0 → 总分差 0.1
        assert abs(high_importance - low_importance - 0.1) < 1e-6

    def test_similarity_dominates_under_default_weights(self):
        from services.long_term_memory_service import score_memory
        now = datetime.utcnow()
        q = _fake_embed("英语")
        unrelated = _fake_embed("语文")

        sim_high = score_memory(
            q, {"embedding": q, "importance": 0.0, "created_at": now},
        )
        sim_low = score_memory(
            q, {"embedding": unrelated, "importance": 1.0, "created_at": now},
        )
        # 0.6 权重的相似度主导:完全不同的向量即使重要度更高也不敌
        assert sim_high > sim_low


# ----------------------------------------------------------------------
# 2. 时间衰减单调性
# ----------------------------------------------------------------------

class TestTimeDecay:

    def test_monotonic_decreasing(self):
        from services.long_term_memory_service import time_decay
        now = datetime.utcnow()
        decays = [
            time_decay(now - timedelta(days=d), decay_half_life_days=30.0)
            for d in (0, 10, 30, 60, 120)
        ]
        # 严格递减
        for a, b in zip(decays, decays[1:]):
            assert a > b
        # 今天≈1,30 天≈0.37
        assert 0.99 <= decays[0] <= 1.0
        assert 0.30 <= decays[2] <= 0.40

    def test_never_reaches_zero(self):
        from services.long_term_memory_service import time_decay
        now = datetime.utcnow()
        d = time_decay(now - timedelta(days=3650), decay_half_life_days=30.0)
        assert d > 0


# ----------------------------------------------------------------------
# 3. 召回 Top-5 + 相似度下限过滤
# ----------------------------------------------------------------------

class TestRetrieval:

    def test_top5_truncation_and_ordering(self, memory_service):
        now = datetime.utcnow()
        # 造 8 条记忆,embedding 与 query 相似度依次下降
        for i in range(8):
            memory_service.add_memory(
                user_id="u1",
                memory_type="preference",
                content=f"记忆{i}: 用户希望安排 {'英语' if i < 6 else '语文'} 家教",
                embedding=_fake_embed(f"英语 家教 安排 时间"),
                importance=0.5,
            )
        # 时间偏移:越新越靠前(用于观察时间衰减对排序的微调)
        with memory_service.db_router.session_manager.session_scope() as s:
            from db.models import LongTermMemory
            rows = s.query(LongTermMemory).all()
            for idx, row in enumerate(rows):
                row.created_at = now - timedelta(days=idx * 5)

        top = memory_service.retrieve_top_k(query="想给孩子安排英语家教", user_id="u1")
        assert len(top) <= 5
        assert len(top) == 5
        # 分数降序
        scores = [m["score"] for m in top]
        assert scores == sorted(scores, reverse=True)

    def test_min_similarity_filters_below_threshold(self, memory_service):
        # 用构造向量精确控制相似度:与 query 同向(cos≈1)命中,正交(cos≈0)被过滤
        q = [1.0, 0.0, 0.0, 0.0]
        related = [1.0, 0.0, 0.0, 0.0]   # cos_sim = 1.0
        orthogonal = [0.0, 1.0, 0.0, 0.0]  # cos_sim = 0.0
        memory_service.add_memory(
            user_id="u1", memory_type="preference",
            content="用户喜欢看电影", embedding=orthogonal,
            importance=1.0,  # 重要度再高也救不回相似度
        )
        memory_service.add_memory(
            user_id="u1", memory_type="preference",
            content="用户希望补英语", embedding=related,
            importance=0.5,
        )
        top = memory_service.retrieve_top_k(
            query="想给孩子安排英语家教", user_id="u1", min_similarity=0.3,
            query_embedding=q,
        )
        contents = [m["content"] for m in top]
        assert "用户希望补英语" in contents
        assert "用户喜欢看电影" not in contents

    def test_no_memory_returns_empty(self, memory_service):
        top = memory_service.retrieve_top_k(query="英语家教", user_id="empty_user")
        assert top == []

    def test_missing_embedding_is_filtered(self, memory_service):
        # 无 embedding 的记忆 cos_sim=0,低于下限被过滤
        memory_service.add_memory(
            user_id="u1", memory_type="preference",
            content="没有向量的旧记忆", embedding=None, importance=1.0,
            auto_embed=False,
        )
        top = memory_service.retrieve_top_k(query="英语家教", user_id="u1")
        assert top == []

    def test_empty_query_returns_empty(self, memory_service):
        memory_service.add_memory(
            user_id="u1", memory_type="preference", content="任意记忆",
            embedding=_fake_embed("任意"), importance=0.5,
        )
        assert memory_service.retrieve_top_k(query="", user_id="u1") == []


# ----------------------------------------------------------------------
# 4. 记忆持久化读写
# ----------------------------------------------------------------------

class TestPersistence:

    def test_add_and_get(self, memory_service):
        mid = memory_service.add_memory(
            user_id="u1", memory_type="appointment",
            content="预约了周三 19:00 英语试讲", embedding=_fake_embed("英语试讲"),
            importance=0.8,
        )
        assert mid > 0
        got = memory_service.get_user_memories("u1")
        assert len(got) == 1
        assert got[0]["content"] == "预约了周三 19:00 英语试讲"
        assert got[0]["memory_type"] == "appointment"
        assert got[0]["importance"] == 0.8
        assert got[0]["embedding"] == _fake_embed("英语试讲")

    def test_update_memory(self, memory_service):
        mid = memory_service.add_memory(
            user_id="u1", memory_type="summary", content="旧摘要", importance=0.5,
        )
        ok = memory_service.update_memory(
            mid, content="新摘要", importance=0.9,
        )
        assert ok
        got = memory_service.get_user_memories("u1")[0]
        assert got["content"] == "新摘要"
        assert got["importance"] == 0.9

    def test_delete_memory(self, memory_service):
        mid = memory_service.add_memory(
            user_id="u1", memory_type="preference", content="待删除",
        )
        assert memory_service.delete_memory(mid)
        assert memory_service.get_user_memories("u1") == []

    def test_filter_by_type(self, memory_service):
        memory_service.add_memory(user_id="u1", memory_type="appointment", content="预约A")
        memory_service.add_memory(user_id="u1", memory_type="summary", content="摘要B")
        appts = memory_service.get_user_memories("u1", memory_type="appointment")
        assert [m["content"] for m in appts] == ["预约A"]


# ----------------------------------------------------------------------
# 5. 60% 窗口滚动摘要
# ----------------------------------------------------------------------

class TestRollupSummary:

    def _build_manager(self, rounds=10, rollup_enabled=True, ratio=0.6,
                       persister=None):
        from agents.context.manager import ContextManager
        from agents.context.contexts.conversation import ConversationContext
        # 构造: conversation(长历史),便于精确控制窗口占用
        conv = ConversationContext()
        session_id = "sess-rollup"
        history = conv.get_history(session_id)
        for i in range(rounds):
            history.add_message(HumanMessage(content=f"用户问题{i}"))
            history.add_message(AIMessage(content=f"顾问回答{i}"))
        mgr = ContextManager(
            [],
            rollup_enabled=rollup_enabled,
            rollup_window_ratio=ratio,
            rollup_rounds=4,
            rollup_persister=persister,
        )
        mgr.session_id = session_id
        mgr.contexts = [conv]
        return mgr, conv

    def test_above_60_percent_triggers_rollup(self):
        mgr, conv = self._build_manager(rounds=20)
        captured = {}
        def persister(user_id, summary_text, session_id=None):
            captured["text"] = summary_text
        mgr._rollup_persister = persister

        # 历史文本很长,max_tokens 设小 → 占用远超 60%
        out = mgr.build(task="当前问题", max_tokens=1000)
        assert mgr.estimate_tokens() <= 1000
        assert mgr.last_rollup_triggered
        # 原文视图不再包含最旧部分;独立显式摘要可以且应包含它。
        assert "用户问题0" not in out.split("【近期滚动摘要】")[0]
        assert "[更早的对话已滚动摘要存档]" in out
        # 最近部分仍保留
        assert "用户问题19" in out
        # 摘要已持久化(有文本)
        assert captured.get("text", "").startswith("早期对话摘要:")

    def test_below_60_percent_does_not_trigger(self):
        mgr, conv = self._build_manager(rounds=2, rollup_enabled=True)
        # 只有 2 轮历史,窗口占用低,max_tokens 给大 → 不触发
        out = mgr.build(task="当前问题", max_tokens=100000)
        assert not mgr.last_rollup_triggered
        assert "用户问题0" in out

    def test_rollup_disabled_by_default(self):
        """默认 rollup_enabled=False,不改变原有行为(无滚动摘要,普通压缩兜底)。"""
        mgr, conv = self._build_manager(rounds=20, rollup_enabled=False)
        out = mgr.build(task="当前问题", max_tokens=200)
        assert mgr.estimate_tokens() <= 200
        assert not mgr.last_rollup_triggered
        # 没有滚动摘要占位(仅可能有普通折叠占位)
        assert "[更早的对话已滚动摘要存档]" not in out

    def test_no_repeat_summary(self):
        """同一会话段只摘要一次(水位防重复)。"""
        # 一段可归档 + 最近4轮;无新段时才不得重复持久化。
        mgr, conv = self._build_manager(rounds=8)
        persisted = []
        def persister(user_id, summary_text, session_id=None):
            persisted.append(summary_text)
        mgr._rollup_persister = persister

        mgr.build(task="q1", max_tokens=400)
        mgr.build(task="q2", max_tokens=400)
        assert conv._rolled_up_count(mgr.session_id) == 8
        # 两次 build,但最旧段只摘要一次 → 只持久化 1 条
        assert len(persisted) == 1

    def test_summary_written_to_long_term_memory(self, memory_service):
        """滚动摘要真正落库到 long_term_memories(summary 类型)。"""
        from agents.context.manager import ContextManager
        from agents.context.contexts.conversation import ConversationContext
        conv = ConversationContext()
        session_id = "sess-persist"
        history = conv.get_history(session_id)
        for i in range(20):
            history.add_message(HumanMessage(content=f"问题{i}"))
            history.add_message(AIMessage(content=f"回答{i}"))

        def persister(user_id, summary_text, session_id=None):
            memory_service.save_rollup_summary(
                user_id=user_id, summary_text=summary_text, session_id=session_id,
            )
        mgr = ContextManager(
            [], rollup_enabled=True, rollup_window_ratio=0.6,
            rollup_rounds=4, rollup_persister=persister,
        )
        mgr.session_id = session_id
        mgr.contexts = [conv]

        mgr.build(task="q", max_tokens=50)
        summaries = memory_service.get_user_memories("default_user", memory_type="summary")
        # 摘要已落库(summary 类型,带 session_id)
        assert summaries, "滚动摘要应写入长期记忆表"
        assert summaries[0]["session_id"] == session_id
        assert "早期对话摘要" in summaries[0]["content"]


# ----------------------------------------------------------------------
# 6. 无记忆 / 无 embedding 时回退原画像摘要(兼容路径)
# ----------------------------------------------------------------------

class TestMemoryContextFallback:

    def _mock_analyzer(self, prefs):
        analyzer = MagicMock()
        analyzer.analyze_user_preferences.return_value = prefs
        return analyzer

    def test_without_task_falls_back_to_profile(self):
        """无 task → 不触发召回,只渲染画像摘要。"""
        from agents.context.contexts.memory import MemoryContext
        analyzer = self._mock_analyzer({
            "favorite_teacher_id": 7, "favorite_subject": "数学",
            "favorite_duration": 60, "total_lessons": 5,
            "last_lesson_date": "2026-09-01",
        })
        ctx = MemoryContext(analyzer=analyzer, memory_service=MagicMock())
        text = ctx.build(_manager(task=""))
        assert "偏好学科: 数学" in text
        assert "相关记忆" not in text  # 未触发召回

    def test_with_task_and_no_memories_falls_back_to_profile(self):
        """有 task 但库里无记忆 → 回退画像摘要,不报错。"""
        from agents.context.contexts.memory import MemoryContext
        analyzer = self._mock_analyzer({
            "favorite_teacher_id": 3, "favorite_subject": "英语",
        })
        empty_svc = MagicMock()
        empty_svc.retrieve_top_k.return_value = []
        ctx = MemoryContext(analyzer=analyzer, memory_service=empty_svc)
        text = ctx.build(_manager(task="想补英语"))
        assert "偏好学科: 英语" in text  # 画像兜底仍在
        assert "相关记忆" not in text

    def test_with_task_and_recalled_memories_injects_both(self, memory_service):
        """有 task + 有记忆 → 召回记忆 + 画像摘要共存。"""
        from agents.context.contexts.memory import MemoryContext
        memory_service.add_memory(
            user_id="default_user", memory_type="preference",
            content="家长偏好每周三晚补英语", embedding=_fake_embed("英语补课"),
            importance=0.8,
        )
        analyzer = self._mock_analyzer({
            "favorite_teacher_id": 3, "favorite_subject": "英语",
        })
        ctx = MemoryContext(analyzer=analyzer, memory_service=memory_service)
        text = ctx.build(_manager(task="给孩子安排英语补课"))
        assert "【相关记忆】" in text
        assert "家长偏好每周三晚补英语" in text
        assert "【用户画像】" in text  # 与召回共存
        assert "偏好学科: 英语" in text

    def test_recall_exception_degrades_gracefully(self):
        """召回抛异常 → 降级为画像摘要,不崩溃。"""
        from agents.context.contexts.memory import MemoryContext
        analyzer = self._mock_analyzer({
            "favorite_teacher_id": 1, "favorite_subject": "物理",
        })
        bad_svc = MagicMock()
        bad_svc.retrieve_top_k.side_effect = RuntimeError("db down")
        ctx = MemoryContext(analyzer=analyzer, memory_service=bad_svc)
        text = ctx.build(_manager(task="补物理"))
        assert "偏好学科: 物理" in text
        assert "相关记忆" not in text

    def test_memory_context_no_profile_no_task_returns_placeholder(self):
        from agents.context.contexts.memory import MemoryContext
        analyzer = self._mock_analyzer(None)
        ctx = MemoryContext(analyzer=analyzer, memory_service=MagicMock())
        text = ctx.build(_manager(task=""))
        assert "暂无用户画像" in text
