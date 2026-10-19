"""
应用程序设置（家教场景）
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AppSettings:
    """应用程序设置"""

    def __init__(self):
        # 数据库
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/smart_appointment.db")

        # 调试
        self.debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # LLM 配置
        self.model_provider: str = os.getenv("MODEL_PROVIDER", "qwen")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "")

        # Embedding 配置
        self.embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "qwen")
        self.embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
        self.embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "")

        # 文本分块配置 (与 config/components.yaml 的 splitter 参数必须一致,
        # 有 tests/test_rag_pipeline_hardening.py 的防漂移断言守卫)
        # 500 中文字 ≈ 500 token < 512, 精排不会截断 chunk 后半段
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

        # 混合检索配置 (hybrid = Dense + BM25 + RRF + ms-marco 精排)
        self.retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | hybrid
        self.rerank_enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() in (
            "1", "true", "yes", "on"
        )
        self.rrf_k: int = int(os.getenv("RRF_K", "60"))
        self.rrf_candidates: int = int(os.getenv("RRF_CANDIDATES", "20"))
        # 精排相关性阈值: 只在本轮真的跑过精排时生效(RRF 兜底路径不适用)。
        # 默认 None = **不设阈值(不过滤)**; 只有显式设置 RAG_MIN_RERANK_SCORE 才生效。
        # 为什么不默认 0.0: bge 的 logit 随 query 口语化程度整体下移, 0.0 不是中性
        # 分界 —— 实测 24 条口语化 query 里 12 条的**正确答案**精排分为负 (多为排名第 1),
        # 阈值 0.0 时 recall@3 从 0.9167 掉到 0.5000。开启前必须按目标模型量纲校准。
        _raw_min_rerank = os.getenv("RAG_MIN_RERANK_SCORE")
        self.min_rerank_score: Optional[float] = None
        if _raw_min_rerank and _raw_min_rerank.strip():
            try:
                self.min_rerank_score = float(_raw_min_rerank)
            except ValueError:
                # 非法值告警并回退 None(不设阈值), 不让配置错误炸掉检索
                logger.warning(
                    "RAG_MIN_RERANK_SCORE=%r 非法, 回退 None (不设阈值)", _raw_min_rerank,
                )

        # 检索前 query 变换(多查询 + 改写): off | rule | llm
        # rule 为默认档(零 LLM 成本);llm 档超时/异常一律回退 rule
        self.query_transform_mode: str = os.getenv("QUERY_TRANSFORM", "rule").strip().lower()
        if self.query_transform_mode not in ("off", "rule", "llm"):
            # 非法值回退 rule, 不让配置错误炸掉检索
            logger.warning(
                "QUERY_TRANSFORM=%r 非法, 回退 rule", self.query_transform_mode,
            )
            self.query_transform_mode = "rule"
        self.query_transform_timeout: float = float(
            os.getenv("QUERY_TRANSFORM_TIMEOUT", "3.0")
        )
        # 变体数上限(含原始 query, 原始 query 恒排第 0 位)
        self.query_transform_max_variants: int = int(
            os.getenv("QUERY_TRANSFORM_MAX_VARIANTS", "3")
        )

    @property
    def mode(self) -> str:
        return "tutor"

    def __repr__(self):
        return f"AppSettings(mode={self.mode}, debug={self.debug})"


settings = AppSettings()