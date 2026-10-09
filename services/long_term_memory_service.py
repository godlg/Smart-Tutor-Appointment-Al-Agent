"""长期记忆服务（分层记忆增强）

职责：
1. 记忆持久化：咨询摘要 / 预约事件 / 偏好统一写入 long_term_memories
2. 召回评分：`score = 0.6*cos_sim + 0.3*time_decay + 0.1*importance` 召回 Top-5
   - 时间衰减 `exp(-days_since / 30)`(半衰期约 20.8 天,30 天后衰减到 0.37)
   - cos_sim 低于 min_similarity(默认 0.3)直接过滤
3. 滚动摘要：把最早一段会话文本折叠成一句摘要,写回 long_term_memories(summary 类型)

五层依赖：agents → services → repositories → db,本模块只依赖 db 层。
embedding 统一复用 services.text_embedding.embed_input()。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from db.db_router import DatabaseRouter

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 召回评分参数（可配,参考 settings.py 风格从 env 读取）
# ----------------------------------------------------------------------
_DEFAULT_WEIGHT_SIM = 0.6     # 向量相似度权重
_DEFAULT_WEIGHT_RECENCY = 0.3  # 时间衰减权重
_DEFAULT_WEIGHT_IMPORTANCE = 0.1  # 重要度权重
_DEFAULT_DECAY_HALF_LIFE_DAYS = 30.0  # exp(-days/30)
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SIMILARITY = 0.3


def _env_float(name: str, default: float) -> float:
    import os
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    import os
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def memory_retrieval_config() -> Dict[str, Any]:
    """召回参数集(env 可配),供 MemoryContext 与测试读取。"""
    return {
        "weight_sim": _env_float("MEMORY_WEIGHT_SIM", _DEFAULT_WEIGHT_SIM),
        "weight_recency": _env_float("MEMORY_WEIGHT_RECENCY", _DEFAULT_WEIGHT_RECENCY),
        "weight_importance": _env_float(
            "MEMORY_WEIGHT_IMPORTANCE", _DEFAULT_WEIGHT_IMPORTANCE
        ),
        "decay_half_life_days": _env_float(
            "MEMORY_DECAY_HALF_LIFE_DAYS", _DEFAULT_DECAY_HALF_LIFE_DAYS
        ),
        "top_k": _env_int("MEMORY_TOP_K", _DEFAULT_TOP_K),
        "min_similarity": _env_float(
            "MEMORY_MIN_SIMILARITY", _DEFAULT_MIN_SIMILARITY
        ),
    }


def _cos_sim(a: List[float], b: List[float]) -> float:
    """余弦相似度;向量为空或维度不一致返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _days_since(dt: Optional[Any], now: Optional[datetime] = None) -> float:
    """把 datetime(可含时区)换算成距 now 的天数(浮点),解析失败按 0 天(最近)处理。"""
    if dt is None:
        return 0.0
    now = now or datetime.utcnow()
    try:
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            dt = dt.replace(tzinfo=None)
        if now.tzinfo is not None and now.utcoffset() is not None:
            now = now.replace(tzinfo=None)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def time_decay(created_at: Optional[Any], decay_half_life_days: float = 30.0) -> float:
    """时间衰减: `exp(-days_since / decay_half_life_days)`, 单调递减、范围 (0,1]。

    - 今天创建的: 1.0(无衰减)
    - 30 天后: ~0.37
    - 越久越接近 0(永不归零,保证最老的记忆仍有一丝权重)
    """
    days = _days_since(created_at)
    return pow(2.718281828459045, -days / max(decay_half_life_days, 1e-6))


def score_memory(
    query_embedding: List[float],
    memory: Dict[str, Any],
    weight_sim: float = _DEFAULT_WEIGHT_SIM,
    weight_recency: float = _DEFAULT_WEIGHT_RECENCY,
    weight_importance: float = _DEFAULT_WEIGHT_IMPORTANCE,
    decay_half_life_days: float = _DEFAULT_DECAY_HALF_LIFE_DAYS,
    now: Optional[datetime] = None,
) -> float:
    """单条记忆的召回分: 0.6*cos_sim + 0.3*time_decay + 0.1*importance。

    缺 embedding 时 cos_sim 按 0 计(仅剩时间/重要度),由调用方决定是否过滤。
    """
    mem_embedding = memory.get("embedding") or []
    sim = _cos_sim(query_embedding, mem_embedding)
    decay = time_decay(memory.get("created_at"), decay_half_life_days)
    importance = memory.get("importance")
    importance = float(importance) if importance is not None else 0.5
    return (
        weight_sim * sim
        + weight_recency * decay
        + weight_importance * importance
    )


class LongTermMemoryService:
    """长期记忆服务：持久化 + 权重召回 + 滚动摘要。"""

    def __init__(self, db_path: str = "sqlite:///data/smart_appointment.db",
                 db_router: Optional[DatabaseRouter] = None):
        self._owns_router = db_router is None
        self.db_router = db_router or DatabaseRouter(db_path)
        self.repo = self.db_router.memory_repo

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        auto_embed: bool = True,
        student_id: Optional[str] = None,
    ) -> int:
        """写入一条长期记忆;embedding 未显式提供且 auto_embed=True 时用 embed_input() 现算。

        显式传 embedding=None + auto_embed=False 可保留"无向量"的记忆(召回时按相似度下限过滤)。
        """
        if embedding is None and auto_embed and content:
            try:
                from .text_embedding import embed_input
                embedding = embed_input(content)
            except Exception as e:
                logger.warning(f"长期记忆 embedding 计算失败,降级为 None: {e}")
        return self.repo.add_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            session_id=session_id,
            student_id=student_id,
        )

    def get_user_memories(
        self, user_id: str, memory_type: Optional[str] = None, limit: Optional[int] = None,
        student_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.repo.get_user_memories(user_id, memory_type, limit, student_id=student_id)

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        return self.repo.update_memory(memory_id, content, embedding, importance)

    def touch_memory(self, memory_id: int) -> bool:
        return self.repo.touch_memory(memory_id)

    def delete_memory(self, memory_id: int) -> bool:
        return self.repo.delete_memory(memory_id)

    # ------------------------------------------------------------------
    # 权重召回
    # ------------------------------------------------------------------

    def retrieve_top_k(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        memory_type: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        now: Optional[datetime] = None,
        student_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按查询召回 Top-k 相关记忆(0.6/0.3/0.1 权重)。

        规则:
        - 每条带 score 字段,按 score 降序
        - cos_sim 低于 min_similarity 的记忆过滤掉(即便时间/重要度再高也不进上下文)
        - embedding 缺失的记忆 cos_sim=0,同样被下限过滤
        - 无记忆 / query 为空 / 向量异常时返回空列表(调用方回退画像摘要)
        """
        config = memory_retrieval_config()
        top_k = top_k or config["top_k"]
        min_similarity = (
            config["min_similarity"]
            if min_similarity is None
            else min_similarity
        )
        if not query or not query.strip():
            return []

        if query_embedding is None:
            try:
                from .text_embedding import embed_input
                query_embedding = embed_input(query)
            except Exception as e:
                logger.warning(f"查询向量计算失败,无法召回: {e}")
                return []
        if not query_embedding:
            return []

        memories = self.repo.get_user_memories(user_id, memory_type, student_id=student_id)
        if not memories:
            return []

        scored = []
        for memory in memories:
            sim = _cos_sim(query_embedding, memory.get("embedding") or [])
            if sim < min_similarity:
                continue
            score = score_memory(
                query_embedding,
                memory,
                weight_sim=config["weight_sim"],
                weight_recency=config["weight_recency"],
                weight_importance=config["weight_importance"],
                decay_half_life_days=config["decay_half_life_days"],
                now=now,
            )
            item = dict(memory)
            item["score"] = score
            item["similarity"] = sim
            scored.append(item)

        scored.sort(key=lambda m: m["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # 滚动摘要
    # ------------------------------------------------------------------

    def save_rollup_summary(
        self,
        user_id: str,
        summary_text: str,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
        student_id: Optional[str] = None,
        auto_embed: bool = True,
    ) -> int:
        """Legacy returns memory id; span mode returns bool after atomic dual archive.

        ConversationService uses span mode with auto_embed=False (no added LLM cost).
        """
        if start_seq is not None or end_seq is not None:
            if not session_id or start_seq is None or end_seq is None:
                raise ValueError("span 摘要要求 session_id、start_seq、end_seq")
            if embedding is None and auto_embed and summary_text:
                try:
                    from .text_embedding import embed_input
                    embedding = embed_input(summary_text)
                except Exception as exc:
                    logger.warning("摘要 embedding 失败,保留文本: %s", exc)
            return self.db_router.conversation_repo.archive_rollup(
                session_id=session_id, user_id=user_id, start_seq=start_seq,
                end_seq=end_seq, content=summary_text, student_id=student_id,
                importance=importance, embedding=embedding,
            )
        return self.add_memory(
            user_id=user_id,
            memory_type="summary",
            content=summary_text,
            embedding=embedding,
            importance=importance,
            session_id=session_id,
            student_id=student_id,
            auto_embed=auto_embed,
        )

    def has_summary_for(
        self, user_id: str, session_id: Optional[str], content_fragment: str
    ) -> bool:
        """摘要去重:同一会话段是否已存在摘要(避免 60% 阈值反复触发重复压)。

        用 session_id + 内容片段做启发式匹配;无 session_id 时按内容片段模糊匹配。
        """
        if not content_fragment:
            return True
        summaries = self.repo.get_user_memories(user_id, "summary")
        for s in summaries:
            if session_id and s.get("session_id") == session_id:
                # 同会话已摘要过 → 视为已处理,不再重复
                return True
            if content_fragment and content_fragment in (s.get("content") or ""):
                return True
        return False
