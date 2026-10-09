"""
应用程序设置（家教场景）
"""

import os
from dotenv import load_dotenv

load_dotenv()


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

        # 混合检索配置 (hybrid = Dense + BM25 + RRF + ms-marco 精排)
        self.retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "dense")  # dense | hybrid
        self.rerank_enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() in (
            "1", "true", "yes", "on"
        )
        self.rrf_k: int = int(os.getenv("RRF_K", "60"))
        self.rrf_candidates: int = int(os.getenv("RRF_CANDIDATES", "20"))

    @property
    def mode(self) -> str:
        return "tutor"

    def __repr__(self):
        return f"AppSettings(mode={self.mode}, debug={self.debug})"


settings = AppSettings()