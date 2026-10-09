"""
ComponentFactory: 读 config/components.yaml → 按 type 实例化六类组件。

设计要点:
1. type → 实现类注册表映射 (dict), 工厂按 YAML 里的 type 查表实例化。
2. YAML 缺失某类 → 用默认实现 (openai_compatible / recursive /
   hybrid / bge / recall_at_k, 与现有代码行为一致)。
3. 懒加载: 组件实例只在首次访问时创建 (create_llm() 等 getter)。
4. 错误清晰: 未知 type 抛 ValueError(带可用 type 列表); 缺参抛清晰提示。
5. 单向依赖: components 层只依赖 services 与 config, 不反向依赖上层。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import BaseEmbedding, BaseEvaluator, BaseLLM, BaseReranker, BaseSplitter, BaseVectorStore, QueryCase
from .evaluators import BgeReranker, NoOpReranker, RecallAtKEvaluator
from .lightweight import OpenAICompatibleEmbedding, OpenAICompatibleLLM, RecursiveSplitter
from .vector_stores import Bm25VectorStore, DenseVectorStore, HybridVectorStore

logger = logging.getLogger(__name__)

# 默认 YAML 路径: 项目根 / config/components.yaml
DEFAULT_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "components.yaml"


# ----------------------------------------------------------------------
# type → 实现类 注册表
# ----------------------------------------------------------------------

LLM_REGISTRY: Dict[str, type] = {"openai_compatible": OpenAICompatibleLLM}
EMBEDDING_REGISTRY: Dict[str, type] = {"openai_compatible": OpenAICompatibleEmbedding}
SPLITTER_REGISTRY: Dict[str, type] = {"recursive": RecursiveSplitter}
VECTOR_STORE_REGISTRY: Dict[str, type] = {
    "dense": DenseVectorStore,
    "bm25": Bm25VectorStore,
    "hybrid": HybridVectorStore,
}
RERANKER_REGISTRY: Dict[str, type] = {"bge": BgeReranker, "none": NoOpReranker}
EVALUATOR_REGISTRY: Dict[str, type] = {"recall_at_k": RecallAtKEvaluator}

# 每类的默认 type (YAML 缺失该 key 时使用)
DEFAULT_TYPES: Dict[str, str] = {
    "llm": "openai_compatible",
    "embedding": "openai_compatible",
    "splitter": "recursive",
    "vector_store": "hybrid",
    "reranker": "bge",
    "evaluator": "recall_at_k",
}

_REGISTRIES: Dict[str, Dict[str, type]] = {
    "llm": LLM_REGISTRY,
    "embedding": EMBEDDING_REGISTRY,
    "splitter": SPLITTER_REGISTRY,
    "vector_store": VECTOR_STORE_REGISTRY,
    "reranker": RERANKER_REGISTRY,
    "evaluator": EVALUATOR_REGISTRY,
}


# ----------------------------------------------------------------------
# YAML 加载
# ----------------------------------------------------------------------

def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    """读 components.yaml; 文件缺失或非 dict 结构时回退空配置 (走默认实现)。"""
    if path is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"components.yaml 解析失败, 回退默认实现: {e}")
        return {}
    components = data.get("components", data)
    return components if isinstance(components, dict) else {}


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

class ComponentFactory:
    """按 YAML 声明创建六类组件实例 (懒加载)。"""

    def __init__(
        self,
        yaml_path: Optional[Path] = None,
        knowledge_service: Any = None,
        evaluator_cases: Optional[List[QueryCase]] = None,
    ) -> None:
        self._config = _load_yaml(yaml_path if yaml_path is not None else DEFAULT_YAML_PATH)
        self._knowledge_service = knowledge_service
        self._evaluator_cases = evaluator_cases or []
        self._instances: Dict[str, Any] = {}

    # ---- 公共 getter ----

    def get_llm(self) -> BaseLLM:
        return self._get("llm")

    def get_embedding(self) -> BaseEmbedding:
        return self._get("embedding")

    def get_splitter(self) -> BaseSplitter:
        return self._get("splitter")

    def get_vector_store(self) -> BaseVectorStore:
        return self._get("vector_store")

    def get_reranker(self) -> BaseReranker:
        return self._get("reranker")

    def get_evaluator(self) -> BaseEvaluator:
        return self._get("evaluator")

    @property
    def config(self) -> Dict[str, Any]:
        """当前生效的组件配置 (含默认值补全)。"""
        return {
            name: {
                "type": self._resolve_type(name),
                **self._raw_params(name),
            }
            for name in _REGISTRIES
        }

    # ---- 内部 ----

    def _get(self, kind: str) -> Any:
        if kind not in self._instances:
            self._instances[kind] = self._create(kind)
        return self._instances[kind]

    def _resolve_type(self, kind: str) -> str:
        spec = self._config.get(kind) or {}
        if isinstance(spec, str):
            return spec
        return spec.get("type", DEFAULT_TYPES[kind])

    def _raw_params(self, kind: str) -> Dict[str, Any]:
        spec = self._config.get(kind) or {}
        if isinstance(spec, str):
            return {}
        return {k: v for k, v in spec.items() if k != "type"}

    def _create(self, kind: str) -> Any:
        type_name = self._resolve_type(kind)
        registry = _REGISTRIES[kind]
        if type_name not in registry:
            raise ValueError(
                f"Unknown {kind} type={type_name!r}. "
                f"Available: {sorted(registry)}. "
                f"Check config/components.yaml (components.{kind}.type)."
            )
        cls = registry[type_name]
        params = self._raw_params(kind)
        logger.debug(f"ComponentFactory: creating {kind}/{type_name} params={params}")
        try:
            if kind == "llm":
                return cls(temperature=params.get("temperature", 0))
            if kind == "embedding":
                return cls()
            if kind == "splitter":
                return cls(
                    chunk_size=params.get("chunk_size", 500),
                    overlap=params.get("overlap", 50),
                )
            if kind == "vector_store":
                return self._create_vector_store(type_name, params)
            if kind == "reranker":
                return cls()
            if kind == "evaluator":
                return cls(cases=self._evaluator_cases, k=params.get("k", 3))
        except TypeError as e:
            raise ValueError(
                f"ComponentFactory: {kind}/{type_name} 实例化参数不匹配 "
                f"(params={params}): {e}"
            )
        raise ValueError(f"ComponentFactory: unhandled kind={kind!r}")

    def _create_vector_store(self, type_name: str, params: Dict[str, Any]) -> Any:
        if self._knowledge_service is None:
            raise ValueError(
                "ComponentFactory: vector_store 需要 knowledge_service 实例, "
                "请在构造时传入 (ComponentFactory(knowledge_service=...))。"
            )
        if type_name == "dense":
            return DenseVectorStore(self._knowledge_service, **params)
        if type_name == "bm25":
            return Bm25VectorStore(**params)
        if type_name == "hybrid":
            return HybridVectorStore(
                self._knowledge_service,
                dense_top_n=params.get("dense_top_n", 20),
                bm25_top_n=params.get("bm25_top_n", 20),
                rrf_k=params.get("rrf_k", 60),
                rerank=params.get("rerank"),
            )
        raise ValueError(f"Unknown vector_store type={type_name!r}")
