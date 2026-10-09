"""
AutoDream 后台任务单元测试

覆盖（对应任务 DoD）：
  1. 会话数 <5 不触发 / ≥5 触发
  2. 距上次 <24h 不触发 / ≥24h 触发
  3. 增量回放：只处理检查点后的新事件
  4. 向量去重：同义候选合并、置信度累加
  5. 冲突降权：矛盾值旧偏好降权、新值正常写入
  6. 幂等：同一批事件重跑不产生重复偏好/日志
  7. 任务锁：重复调度只跑一次

Embedding 用确定性 fake（sha256→16 维），不依赖远程 API。
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fake_embed(text: str) -> List[float]:
    """确定性 fake embedding：相同文本→完全相同向量（相似度 1.0），
    不同文本→近正交（相似度≈0），用于区分"同义合并"与"冲突降权"。"""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    import random
    rng = random.Random(seed)
    vec = [rng.uniform(-1, 1) for _ in range(64)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """全局替换 embed_input，避免真实 API 调用。"""
    from services import autodream_service, text_embedding
    monkeypatch.setattr(autodream_service, "embed_input", _fake_embed)
    monkeypatch.setattr(text_embedding, "embed_input", _fake_embed)


@pytest.fixture
def env(tmp_path):
    """临时 SQLite 库 + AutoDreamService（不碰生产库）。"""
    db_path = f"sqlite:///{tmp_path / 'autodream_test.db'}"
    from db.db_router import DatabaseRouter
    from services.autodream_service import AutoDreamService
    db = DatabaseRouter(db_path)
    svc = AutoDreamService(db_path=db_path)
    return {"db": db, "svc": svc, "db_path": db_path}


def _record(db, user_id, session_id, action_type="lesson", action_data=None,
            teacher_id=None):
    return db.user_behavior_repo.record_behavior(
        user_id=user_id,
        action_type=action_type,
        action_data=action_data or {},
        teacher_id=teacher_id,
        session_id=session_id,
    )


def _record_lesson(db, user_id, session_id, subject="数学", hour=10,
                   duration=60, teacher_id=1):
    return _record(
        db, user_id, session_id,
        action_data={
            "subject": subject,
            "scheduled_at": f"2026-09-22 {hour:02d}:00:00",
            "duration_minutes": duration,
        },
        teacher_id=teacher_id,
    )


def _prefs(db, user_id):
    return db.user_behavior_repo.get_user_preferences(user_id)


def _log_count(db, user_id):
    from db.models import AutoDreamChangeLog
    with db.session_manager.session_scope() as s:
        return s.query(AutoDreamChangeLog).filter(
            AutoDreamChangeLog.user_id == user_id
        ).count()


# ----------------------------------------------------------------------
# 1. 触发条件：会话数
# ----------------------------------------------------------------------

class TestSessionCountTrigger:
    def test_less_than_5_sessions_does_not_trigger(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(4):
            _record_lesson(db, "u1", f"s{i}")
        result = svc.run_for_user("u1")
        assert result["triggered"] is False
        assert result["skipped"] == "condition"
        assert "会话数" in result["reason"]

    def test_5_sessions_triggers(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")
        result = svc.run_for_user("u1")
        assert result["triggered"] is True
        assert result["preferences_written"] >= 4  # subject/time/duration/teacher

    def test_duplicate_session_ids_count_once(self, env):
        """同一 session_id 多行为只算 1 个会话。"""
        db, svc = env["db"], env["svc"]
        for i in range(8):
            _record_lesson(db, "u1", "same_session")  # 8 条行为，但只有 1 个会话
        result = svc.run_for_user("u1")
        assert result["triggered"] is False
        assert result["skipped"] == "condition"


# ----------------------------------------------------------------------
# 2. 触发条件：距上次 ≥24h
# ----------------------------------------------------------------------

class TestIntervalTrigger:
    def test_within_24h_does_not_trigger(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")
        # 首次回放
        svc.run_for_user("u1")
        # 立即再跑 → 距上次 <24h，不触发
        result = svc.run_for_user("u1")
        assert result["triggered"] is False
        assert "24" in result["reason"]

    def test_after_24h_triggers(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")
        svc.run_for_user("u1")
        # 把检查点时间拨到 25h 前
        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)
        # 加一个新事件，验证增量回放
        _record_lesson(db, "u1", "s5", subject="英语", hour=19)
        result = svc.run_for_user("u1")
        assert result["triggered"] is True
        assert result["processed_events"] == 1  # 只回放新事件


# ----------------------------------------------------------------------
# 3. 增量回放：检查点水位
# ----------------------------------------------------------------------

class TestIncrementalReplay:
    def test_only_new_events_after_checkpoint(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")
        first = svc.run_for_user("u1")
        assert first["processed_events"] == 5
        first_pref_count = len(_prefs(db, "u1"))
        first_log_count = _log_count(db, "u1")

        # 拨快到 ≥24h，再加 2 个新事件
        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)

        _record_lesson(db, "u1", "s5", subject="英语", hour=19)
        _record_lesson(db, "u1", "s6", subject="英语", hour=19)
        second = svc.run_for_user("u1")
        assert second["triggered"] is True
        assert second["processed_events"] == 2  # 只处理检查点后新事件
        assert len(_prefs(db, "u1")) >= first_pref_count
        assert _log_count(db, "u1") > first_log_count


# ----------------------------------------------------------------------
# 4. 向量去重：同义合并 + 置信度累加
# ----------------------------------------------------------------------

class TestVectorDedup:
    def test_synonym_merges_and_accumulates(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}", subject="数学")
        svc.run_for_user("u1")

        # 拨快到 ≥24h，新事件带同义学科（"数学" 与 "数学" embedding 完全相同 → 命中合并）
        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)

        _record_lesson(db, "u1", "s5", subject="数学")
        result = svc.run_for_user("u1")
        assert result["triggered"] is True

        prefs = _prefs(db, "u1")
        math_prefs = [p for p in prefs if p["preference_type"] == "subject"
                      and p["preference_value"] == "数学"]
        assert len(math_prefs) == 1, "同义候选应合并为一条，而非新增"
        assert math_prefs[0]["confidence_score"] == 6  # 首次 5 次 +1 次合并 = 6

    def test_confidence_caps_at_max(self, env):
        """上限 CONFIDENCE_MAX=10，多轮合并不无限累加。"""
        from services.autodream_service import CONFIDENCE_MAX
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}", subject="数学")
        svc.run_for_user("u1")

        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)

        for i in range(CONFIDENCE_MAX + 3):  # 跑多轮合并
            with db.session_manager.session_scope() as s:
                cp = s.query(AutoDreamCheckpoint).filter(
                    AutoDreamCheckpoint.user_id == "u1"
                ).first()
                cp.last_run_at = datetime.utcnow() - timedelta(hours=25)
            _record_lesson(db, "u1", f"s{i+5}", subject="数学")
            svc.run_for_user("u1")

        math_prefs = [p for p in _prefs(db, "u1")
                      if p["preference_type"] == "subject"
                      and p["preference_value"] == "数学"]
        assert math_prefs[0]["confidence_score"] == CONFIDENCE_MAX


# ----------------------------------------------------------------------
# 5. 冲突降权：矛盾值旧偏好降权、新值正常写入
# ----------------------------------------------------------------------

class TestConflictDowngrade:
    def test_conflict_downgrades_old_value(self, env):
        db, svc = env["db"], env["svc"]
        # 5 个会话全部上午 → 时段偏好"上午"
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}", hour=10)
        svc.run_for_user("u1")
        morning = [p for p in _prefs(db, "u1")
                   if p["preference_type"] == "time_period"]
        assert morning and morning[0]["preference_value"] == "上午"
        assert morning[0]["confidence_score"] == 5

        # 拨快到 ≥24h，新事件时段"晚上"（矛盾值）
        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)

        _record_lesson(db, "u1", "s5", hour=19)
        result = svc.run_for_user("u1")
        assert result["triggered"] is True
        assert result["actions"]["downgrade"] >= 1, "应发生冲突降权"

        prefs = _prefs(db, "u1")
        morning_now = [p for p in prefs
                       if p["preference_type"] == "time_period"
                       and p["preference_value"] == "上午"]
        evening_now = [p for p in prefs
                       if p["preference_type"] == "time_period"
                       and p["preference_value"] == "晚上"]
        # 旧值降权（5→4），新值正常写入
        assert morning_now[0]["confidence_score"] == 4, "旧值应降权"
        assert evening_now and evening_now[0]["confidence_score"] == 1, "新值正常写入"

        # 日志留痕含 downgrade
        from db.models import AutoDreamChangeLog
        with db.session_manager.session_scope() as s:
            downgrades = s.query(AutoDreamChangeLog).filter(
                AutoDreamChangeLog.user_id == "u1",
                AutoDreamChangeLog.action == "downgrade",
            ).all()
        assert downgrades, "应留下 downgrade 变更日志"


# ----------------------------------------------------------------------
# 6. 幂等：重跑同一批不产生重复偏好/日志
# ----------------------------------------------------------------------

class TestIdempotency:
    def test_replay_same_batch_no_duplicates(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")
        svc.run_for_user("u1")

        pref_count = len(_prefs(db, "u1"))
        log_count = _log_count(db, "u1")

        # 立即重跑（<24h，被触发条件拦住，不处理）
        result = svc.run_for_user("u1")
        assert result["triggered"] is False
        assert len(_prefs(db, "u1")) == pref_count
        assert _log_count(db, "u1") == log_count

        # 拨快 25h 重跑：检查点已推进，无新事件 → 仍无重复
        from db.models import AutoDreamCheckpoint
        with db.session_manager.session_scope() as s:
            cp = s.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == "u1"
            ).first()
            cp.last_run_at = datetime.utcnow() - timedelta(hours=25)
        result = svc.run_for_user("u1")
        assert result["triggered"] is True
        assert result["processed_events"] == 0, "检查点之后无新事件"
        assert len(_prefs(db, "u1")) == pref_count, "不产生重复偏好"
        assert _log_count(db, "u1") == log_count, "不产生重复日志"


# ----------------------------------------------------------------------
# 7. 任务锁：重复调度只跑一次
# ----------------------------------------------------------------------

class TestTaskLock:
    def test_second_call_skipped_while_locked(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")

        # 手动持有锁 → run_for_user 应跳过
        from services.autodream_service import AutoDreamService
        token = "manual-token"
        assert svc._acquire_lock("u1", token) is True
        result = svc.run_for_user("u1")
        assert result["skipped"] == "locked"
        assert result["triggered"] is False
        # 没有写任何偏好
        assert _prefs(db, "u1") == []

        # 释放后再跑 → 正常触发
        svc._release_lock("u1", token)
        result = svc.run_for_user("u1")
        assert result["triggered"] is True

    def test_expired_lock_can_be_taken_over(self, env):
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")

        # 制造一个已过期锁
        from datetime import timedelta
        svc._acquire_lock("u1", "stale-token")
        from db.models import AutoDreamLock
        with db.session_manager.session_scope() as s:
            lock = s.query(AutoDreamLock).filter(
                AutoDreamLock.user_id == "u1"
            ).first()
            lock.expires_at = datetime.utcnow() - timedelta(seconds=1)

        result = svc.run_for_user("u1")
        assert result["triggered"] is True, "过期锁应可抢占"
        # 结束后锁应释放
        from db.models import AutoDreamLock
        with db.session_manager.session_scope() as s:
            assert s.query(AutoDreamLock).filter(
                AutoDreamLock.user_id == "u1"
            ).first() is None

    def test_lock_released_after_failure(self, env):
        """异常路径（取事件抛错）也要 finally 释放锁。"""
        db, svc = env["db"], env["svc"]
        for i in range(5):
            _record_lesson(db, "u1", f"s{i}")

        from services import autodream_service

        def _boom(*a, **k):
            raise RuntimeError("模拟取事件崩溃")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(autodream_service.AutoDreamService,
                            "_get_new_behaviors", _boom)
        try:
            with pytest.raises(RuntimeError):
                svc.run_for_user("u1")
        finally:
            monkeypatch.undo()

        from db.models import AutoDreamLock
        with db.session_manager.session_scope() as s:
            assert s.query(AutoDreamLock).filter(
                AutoDreamLock.user_id == "u1"
            ).first() is None, "失败后锁应释放"


# ----------------------------------------------------------------------
# 8. 抽取维度
# ----------------------------------------------------------------------

class TestExtraction:
    def test_extract_lesson_candidates(self, env):
        svc = env["svc"]
        behavior = {
            "action_type": "lesson",
            "action_data": {
                "subject": "数学",
                "scheduled_at": "2026-09-22 10:00:00",
                "duration_minutes": 90,
            },
            "teacher_id": 3,
        }
        cands = svc.extract_candidates(behavior)
        types = {c["preference_type"]: c["preference_value"] for c in cands}
        assert types["subject"] == "数学"
        assert types["time_period"] == "上午"
        assert types["duration"] == "90"
        assert types["teacher"] == "3"

    def test_extract_consultation_subject_fallback(self, env):
        svc = env["svc"]
        behavior = {
            "action_type": "consultation",
            "action_data": {"query": "想给孩子补英语"},
            "teacher_id": None,
        }
        cands = svc.extract_candidates(behavior)
        subjects = [c for c in cands if c["preference_type"] == "subject"]
        assert subjects and subjects[0]["preference_value"] == "英语"
