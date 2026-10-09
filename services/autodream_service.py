"""
AutoDream 后台任务：用户行为增量回放与长期偏好更新

叙事：行为事件 → 增量回放 → 结构化抽取 → 向量去重 → 冲突降权 →
      置信度更新 → 长期偏好，用任务锁 / 检查点 / 变更日志保证幂等重试。

触发条件（每 user 维度）：
  - 该 user 会话数（session_id 去重）≥ 5
  - 距上次成功回放 ≥ 24h（检查点决定，首次视为满足）

流程：
  1. 检查任务锁（autodream_locks，带 TTL），已锁则跳过 → 并发只跑一次
  2. 检查触发条件，不满足直接跳过
  3. 增量取「上次检查点 last_event_id 之后」的新行为事件
  4. 结构化抽取候选偏好（学科 / 时段 / 时长 / 老师）
  5. 向量去重：与同类型已有偏好用 embed_input() 算相似度，≥0.85 合并（置信度+1），否则新增
  6. 冲突降权：同类型矛盾值不覆盖旧值，旧值 confidence-1（下限 1），新值正常写入
  7. 全部写回 user_preferences 并写 autodream_change_logs 留痕
  8. 回放成功才推进检查点水位；失败不推进 → 下次重试（幂等来源）
  9. finally 释放任务锁
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from db.db_router import DatabaseRouter
from db.models import (
    AutoDreamChangeLog,
    AutoDreamCheckpoint,
    AutoDreamLock,
    UserBehavior,
    UserPreference,
)
from services.text_embedding import embed_input

logger = logging.getLogger(__name__)

# 触发条件常量
MIN_SESSIONS = 5          # 会话数下限（session_id 去重）
MIN_INTERVAL_HOURS = 24   # 距上次成功回放的最小间隔
SIMILARITY_THRESHOLD = 0.85  # 向量去重相似度阈值
CONFIDENCE_MAX = 10       # 置信度上限
CONFIDENCE_MIN = 1        # 置信度下限
LOCK_TTL_SECONDS = 3600   # 任务锁 TTL（异常退出后超时自动失效）

# 偏好类型维度（与 pattern_analyzer / preference_manager 对齐）
PREFERENCE_TYPES = ("subject", "time_period", "duration", "teacher")


class AutoDreamService:
    """AutoDream 后台任务服务"""

    def __init__(self, db_path: str = 'sqlite:///data/smart_appointment.db'):
        self.db = DatabaseRouter(db_path)
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 触发条件
    # ------------------------------------------------------------------

    def _count_sessions(self, user_id: str) -> int:
        """按 session_id 去重统计会话数（None/空串忽略）。"""
        with self.db.session_manager.session_scope() as session:
            session_ids = {
                b.session_id
                for b in session.query(UserBehavior.session_id).filter(
                    UserBehavior.user_id == user_id
                ).all()
                if b.session_id
            }
            return len(session_ids)

    def _get_checkpoint(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session_manager.session_scope() as session:
            cp = session.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == user_id
            ).first()
            if cp is None:
                return None
            return {
                "id": cp.id,
                "user_id": cp.user_id,
                "last_event_id": cp.last_event_id or 0,
                "last_run_at": cp.last_run_at,
                "total_replays": cp.total_replays or 0,
            }

    def should_run(self, user_id: str) -> Dict[str, Any]:
        """
        判断该 user 是否满足触发条件。

        Returns:
            {"should_run": bool, "reason": str, "sessions": int, "hours_since_last_run": float|None}
        """
        sessions = self._count_sessions(user_id)
        if sessions < MIN_SESSIONS:
            return {
                "should_run": False,
                "reason": f"会话数 {sessions} < {MIN_SESSIONS}",
                "sessions": sessions,
                "hours_since_last_run": None,
            }

        cp = self._get_checkpoint(user_id)
        if cp is None or cp["last_run_at"] is None:
            return {
                "should_run": True,
                "reason": "首次回放（无检查点）",
                "sessions": sessions,
                "hours_since_last_run": None,
            }

        hours = (datetime.utcnow() - cp["last_run_at"]).total_seconds() / 3600.0
        if hours < MIN_INTERVAL_HOURS:
            return {
                "should_run": False,
                "reason": f"距上次回放 {hours:.1f}h < {MIN_INTERVAL_HOURS}h",
                "sessions": sessions,
                "hours_since_last_run": hours,
            }
        return {
            "should_run": True,
            "reason": f"距上次回放 {hours:.1f}h ≥ {MIN_INTERVAL_HOURS}h",
            "sessions": sessions,
            "hours_since_last_run": hours,
        }

    # ------------------------------------------------------------------
    # 任务锁
    # ------------------------------------------------------------------

    def _acquire_lock(self, user_id: str, token: str) -> bool:
        """尝试获取任务锁。已存在未过期锁则失败；过期锁可抢占。"""
        now = datetime.utcnow()
        with self.db.session_manager.session_scope() as session:
            lock = session.query(AutoDreamLock).filter(
                AutoDreamLock.user_id == user_id
            ).first()
            if lock is None:
                session.add(AutoDreamLock(
                    user_id=user_id,
                    lock_token=token,
                    locked_at=now,
                    expires_at=now + timedelta(seconds=LOCK_TTL_SECONDS),
                ))
                return True
            if lock.expires_at is not None and lock.expires_at < now:
                # 过期锁：抢占
                lock.lock_token = token
                lock.locked_at = now
                lock.expires_at = now + timedelta(seconds=LOCK_TTL_SECONDS)
                return True
            return False

    def _release_lock(self, user_id: str, token: str) -> None:
        """释放任务锁（仅释放自己持有的）。"""
        with self.db.session_manager.session_scope() as session:
            lock = session.query(AutoDreamLock).filter(
                AutoDreamLock.user_id == user_id
            ).first()
            if lock and lock.lock_token == token:
                session.delete(lock)

    # ------------------------------------------------------------------
    # 增量取事件（水位：user_behaviors.id）
    # ------------------------------------------------------------------

    def _get_new_behaviors(self, user_id: str, last_event_id: int) -> List[Dict[str, Any]]:
        with self.db.session_manager.session_scope() as session:
            rows = (
                session.query(UserBehavior)
                .filter(
                    UserBehavior.user_id == user_id,
                    UserBehavior.id > last_event_id,
                )
                .order_by(UserBehavior.id.asc())
                .all()
            )
            return [
                {
                    "id": b.id,
                    "action_type": b.action_type,
                    "action_data": b.action_data or {},
                    "teacher_id": b.teacher_id,
                    "session_id": b.session_id,
                    "created_at": b.created_at,
                }
                for b in rows
            ]

    # ------------------------------------------------------------------
    # 结构化抽取（对齐 pattern_analyzer 的维度）
    # ------------------------------------------------------------------

    def extract_candidates(self, behavior: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从单个行为事件抽取候选偏好。

        Returns:
            [{"preference_type": str, "preference_value": str}, ...]
        """
        candidates: List[Dict[str, str]] = []
        action_data = behavior.get("action_data") or {}
        action_type = behavior.get("action_type")

        # 学科
        subject = action_data.get("subject")
        if subject:
            candidates.append({"preference_type": "subject", "preference_value": str(subject)})

        # 时段（scheduled_at → 上午/下午/晚上）
        scheduled_at = action_data.get("scheduled_at")
        if scheduled_at:
            try:
                dt = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
                hour = dt.hour
                if 6 <= hour < 12:
                    period = "上午"
                elif 12 <= hour < 18:
                    period = "下午"
                else:
                    period = "晚上"
                candidates.append({"preference_type": "time_period", "preference_value": period})
            except (ValueError, TypeError):
                logger.debug(f"AutoDream: 解析 scheduled_at 失败: {scheduled_at}")

        # 时长
        duration = action_data.get("duration_minutes")
        if duration is not None:
            candidates.append({"preference_type": "duration", "preference_value": str(duration)})

        # 老师（teacher_id 列优先，其次 action_data）
        teacher_id = behavior.get("teacher_id")
        if teacher_id is None:
            teacher_id = action_data.get("teacher_id")
        if teacher_id is not None:
            candidates.append({"preference_type": "teacher", "preference_value": str(teacher_id)})

        # 咨询行为里的意向学科（action_type=consultation 且无 subject 时兜底）
        if action_type == "consultation" and not any(
            c["preference_type"] == "subject" for c in candidates
        ):
            query = action_data.get("query") or action_data.get("question") or ""
            for kw in ("数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治"):
                if kw in query:
                    candidates.append({"preference_type": "subject", "preference_value": kw})
                    break

        return candidates

    # ------------------------------------------------------------------
    # 向量去重 / 冲突降权 / 置信度更新（写回偏好 + 变更日志）
    # ------------------------------------------------------------------

    def _get_existing_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        with self.db.session_manager.session_scope() as session:
            rows = session.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).all()
            return [
                {"id": p.id, "preference_type": p.preference_type,
                 "preference_value": p.preference_value,
                 "confidence_score": p.confidence_score}
                for p in rows
            ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度（embed_input 产出未归一化向量时兜底归一化）。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _find_similar_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: str,
    ) -> Optional[Dict[str, Any]]:
        """
        向量去重：与同类型已有偏好算 embedding 相似度，≥0.85 视为同义。

        Returns:
            命中的已有偏好 dict；无则 None。
        """
        existing = [
            p for p in self._get_existing_preferences(user_id)
            if p["preference_type"] == preference_type
        ]
        if not existing:
            return None

        try:
            new_vec = embed_input(f"{preference_type}:{preference_value}")
        except Exception as e:
            logger.warning(f"AutoDream: embed_input 失败({e})，跳过向量去重直接新增")
            return None

        best = None
        best_sim = -1.0
        for p in existing:
            try:
                old_vec = embed_input(f"{p['preference_type']}:{p['preference_value']}")
            except Exception as e:
                logger.warning(f"AutoDream: embed_input 失败({e})，跳过该候选")
                continue
            sim = self._cosine_similarity(new_vec, old_vec)
            if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
                best_sim = sim
                best = p
        return best

    def _downgrade_confidence(self, user_id: str, preference_id: int, event_id: int,
                              preference_type: str, preference_value: str,
                              detail: Dict[str, Any]) -> None:
        """冲突降权：旧值 confidence -1（下限 1），并写 downgrade 日志。"""
        with self.db.session_manager.session_scope() as session:
            p = session.query(UserPreference).filter(
                UserPreference.id == preference_id
            ).first()
            if p and p.confidence_score > CONFIDENCE_MIN:
                old_score = p.confidence_score
                p.confidence_score -= 1
                p.last_updated = datetime.utcnow()
                detail = dict(detail)
                detail["old_confidence"] = old_score
                detail["new_confidence"] = p.confidence_score
                session.add(AutoDreamChangeLog(
                    user_id=user_id,
                    event_id=event_id,
                    action="downgrade",
                    preference_type=preference_type,
                    preference_value=preference_value,
                    detail=detail,
                ))
                logger.info(
                    f"AutoDream: 冲突降权 user={user_id} {preference_type}={preference_value} "
                    f"{old_score}→{p.confidence_score}"
                )

    def _merge_preference(self, user_id: str, matched: Dict[str, Any], event_id: int,
                          new_value: str, similarity: float) -> None:
        """合并：同义候选命中已有偏好，置信度 +1（上限 CONFIDENCE_MAX）。"""
        with self.db.session_manager.session_scope() as session:
            p = session.query(UserPreference).filter(
                UserPreference.id == matched["id"]
            ).first()
            if p:
                old_score = p.confidence_score
                p.confidence_score = min(CONFIDENCE_MAX, p.confidence_score + 1)
                p.last_updated = datetime.utcnow()
                session.add(AutoDreamChangeLog(
                    user_id=user_id,
                    event_id=event_id,
                    action="merge",
                    preference_type=p.preference_type,
                    preference_value=p.preference_value,
                    detail={
                        "similarity": round(similarity, 4),
                        "new_value": new_value,
                        "old_confidence": old_score,
                        "new_confidence": p.confidence_score,
                    },
                ))
                logger.info(
                    f"AutoDream: 合并 user={user_id} {p.preference_type} "
                    f"{p.preference_value} 置信度 {old_score}→{p.confidence_score}"
                )

    def _create_preference(self, user_id: str, preference_type: str, preference_value: str,
                           event_id: int, similarity: Optional[float]) -> None:
        """新增：无同义偏好时写入新偏好（confidence=1）+ new 日志。"""
        with self.db.session_manager.session_scope() as session:
            p = UserPreference(
                user_id=user_id,
                preference_type=preference_type,
                preference_value=preference_value,
                confidence_score=1,
            )
            session.add(p)
            session.flush()
            session.add(AutoDreamChangeLog(
                user_id=user_id,
                event_id=event_id,
                action="new",
                preference_type=preference_type,
                preference_value=preference_value,
                detail={"similarity": similarity, "new_confidence": 1},
            ))
            logger.info(f"AutoDream: 新增偏好 user={user_id} {preference_type}={preference_value}")

    def apply_candidate(self, user_id: str, candidate: Dict[str, str], event_id: int) -> str:
        """
        应用单个候选偏好：向量去重 → 命中合并，否则冲突降权/新增。

        Returns:
            "merge" / "new" / "downgrade"
        """
        ptype = candidate["preference_type"]
        pvalue = candidate["preference_value"]

        matched = self._find_similar_preference(user_id, ptype, pvalue)
        if matched is not None:
            sim = self._similarity_of(matched, ptype, pvalue)
            self._merge_preference(user_id, matched, event_id, pvalue, sim)
            return "merge"

        # 未命中同义 → 检查同类型是否已有不同值（矛盾 → 冲突降权）
        conflicting = [
            p for p in self._get_existing_preferences(user_id)
            if p["preference_type"] == ptype and p["preference_value"] != pvalue
        ]
        for p in conflicting:
            self._downgrade_confidence(
                user_id, p["id"], event_id, ptype, p["preference_value"],
                detail={"conflict_with": pvalue},
            )
        self._create_preference(user_id, ptype, pvalue, event_id, None)
        return "new" if not conflicting else "downgrade"

    def _similarity_of(self, matched: Dict[str, Any], ptype: str, pvalue: str) -> float:
        """重算命中偏好与候选的相似度（供日志 detail 用）。"""
        try:
            a = embed_input(f"{ptype}:{pvalue}")
            b = embed_input(f"{matched['preference_type']}:{matched['preference_value']}")
            return self._cosine_similarity(a, b)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def _advance_checkpoint(self, user_id: str, last_event_id: int) -> None:
        """回放成功才推进水位（幂等的来源）。"""
        now = datetime.utcnow()
        with self.db.session_manager.session_scope() as session:
            cp = session.query(AutoDreamCheckpoint).filter(
                AutoDreamCheckpoint.user_id == user_id
            ).first()
            if cp is None:
                session.add(AutoDreamCheckpoint(
                    user_id=user_id,
                    last_event_id=last_event_id,
                    last_run_at=now,
                    total_replays=1,
                ))
            else:
                cp.last_event_id = max(cp.last_event_id, last_event_id)
                cp.last_run_at = now
                cp.total_replays = (cp.total_replays or 0) + 1

    def run_for_user(self, user_id: str) -> Dict[str, Any]:
        """
        对单个 user 执行一次 AutoDream 回放。

        Returns:
            {"user_id", "triggered": bool, "reason": str,
             "processed_events": int, "preferences_written": int,
             "actions": {merge/new/downgrade 计数}, "skipped": str|None}
        """
        token = uuid.uuid4().hex
        result: Dict[str, Any] = {
            "user_id": user_id,
            "triggered": False,
            "reason": "",
            "processed_events": 0,
            "preferences_written": 0,
            "actions": {"merge": 0, "new": 0, "downgrade": 0},
            "skipped": None,
        }

        # 1) 任务锁：已锁则跳过（重复调度只跑一次）
        if not self._acquire_lock(user_id, token):
            result["skipped"] = "locked"
            result["reason"] = "任务锁已持有，跳过"
            return result

        try:
            # 2) 触发条件
            check = self.should_run(user_id)
            if not check["should_run"]:
                result["skipped"] = "condition"
                result["reason"] = check["reason"]
                return result

            # 3) 增量取事件
            cp = self._get_checkpoint(user_id)
            last_event_id = cp["last_event_id"] if cp else 0
            behaviors = self._get_new_behaviors(user_id, last_event_id)
            result["processed_events"] = len(behaviors)
            if not behaviors:
                # 无新事件也推进检查点（更新 last_run_at，避免 24h 后仍卡在无事件重试）
                self._advance_checkpoint(user_id, last_event_id)
                result["triggered"] = True
                result["reason"] = "无新事件，已更新检查点时间"
                return result

            # 4) 回放 + 抽取 + 去重/降权/合并
            max_event_id = last_event_id
            written = 0
            for b in behaviors:
                max_event_id = max(max_event_id, b["id"])
                for cand in self.extract_candidates(b):
                    action = self.apply_candidate(user_id, cand, b["id"])
                    result["actions"][action] += 1
                    written += 1

            # 5) 成功 → 推进水位
            self._advance_checkpoint(user_id, max_event_id)
            result["triggered"] = True
            result["preferences_written"] = written
            result["reason"] = (
                f"回放 {len(behaviors)} 个事件，抽取写入 {written} 条偏好"
            )
            return result
        except Exception as e:
            logger.error(f"AutoDream: 回放失败 user={user_id}: {e}")
            result["reason"] = f"失败：{e}"
            raise
        finally:
            # 6) 异常也释放锁（异常退出不阻塞下次调度）
            self._release_lock(user_id, token)

    def run_once(self) -> Dict[str, Any]:
        """
        单次运行入口：扫描所有有行为数据的 user，逐个执行回放。

        供测试 / API 手动触发；调度器每 30 分钟调用一次。
        """
        with self.db.session_manager.session_scope() as session:
            user_ids = [
                r[0] for r in session.query(UserBehavior.user_id).distinct().all()
            ]
        summary = {"total_users": len(user_ids), "processed": 0, "skipped": 0, "results": []}
        for uid in user_ids:
            try:
                res = self.run_for_user(uid)
            except Exception:
                summary["skipped"] += 1
                continue
            summary["results"].append(res)
            if res.get("triggered"):
                summary["processed"] += 1
            else:
                summary["skipped"] += 1
        return summary

    # ------------------------------------------------------------------
    # 调度器（照抄 recommendation_service 的 schedule + threading 模式）
    # ------------------------------------------------------------------

    def start_scheduler(self, interval_minutes: int = 30) -> bool:
        if self.is_running:
            return False
        try:
            import schedule

            schedule.every(interval_minutes).minutes.do(self.run_once)
            self.is_running = True

            def run_scheduler():
                logger.info(f"AutoDream 调度器已启动（每 {interval_minutes} 分钟）")
                import time
                while self.is_running:
                    schedule.run_pending()
                    time.sleep(60)
                logger.info("AutoDream 调度器已停止")

            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            return True
        except Exception as e:
            logger.error(f"AutoDream: 启动调度器失败: {e}")
            return False

    def stop_scheduler(self) -> bool:
        try:
            self.is_running = False
            import schedule
            schedule.clear()
            return True
        except Exception as e:
            logger.error(f"AutoDream: 停止调度器失败: {e}")
            return False
