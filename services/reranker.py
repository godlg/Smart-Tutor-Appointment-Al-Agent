# services/reranker.py
"""本地 cross-encoder 精排器（ONNX, CPU 推理）。

- cross-encoder: query 与 doc 拼成一对过 transformer 打分, 比 bi-encoder 准但慢,
  因此只在 RRF 融合后的 top-N 候选上精排(几十对, 每对 ~5ms)。
- 候选模型按优先级自动探测(cache/ 下已下载者优先), 均支持中文精排:
    1. bge-reranker-base      (Xenova ONNX quantized, 多语言, 中文效果最佳)
    2. ms-marco-MiniLM-L-6-v2 (英文模型, 中文场景区分度差, 仅作保留候选)
- 模型缺失/加载失败: available() 返回 False, 由调用方静默回退 RRF 结果。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 模型目录: 项目根/cache/<name>/ (含 onnx/model*.onnx + tokenizer.json 等)
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# 候选模型 (目录名, onnx 文件名, 显示名) — 按优先级
MODEL_CANDIDATES: List[Dict[str, str]] = [
    {
        "name": "bge-reranker-base",
        "dir": _CACHE_DIR / "bge-reranker-base",
        "file": _CACHE_DIR / "bge-reranker-base" / "onnx" / "model_quantized.onnx",
    },
    {
        "name": "ms-marco-MiniLM-L-6-v2",
        "dir": _CACHE_DIR / "ms-marco-onnx",
        "file": _CACHE_DIR / "ms-marco-onnx" / "onnx" / "model.onnx",
    },
]

# 进程内单例(懒加载): 避免每次查询重复加载几百 MB 模型
_session: Any = None
_tokenizer: Any = None
_active_model: Optional[str] = None
_load_attempted = False


def _load() -> bool:
    """加载 tokenizer + onnx session(取第一个模型文件存在的候选); 失败置标记不再重试。"""
    global _session, _tokenizer, _active_model, _load_attempted
    if _load_attempted:
        return _session is not None and _tokenizer is not None
    _load_attempted = True
    try:
        import numpy as np  # noqa: F401  (onnxruntime 需要)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        for cand in MODEL_CANDIDATES:
            if not cand["file"].exists():
                continue
            _tokenizer = AutoTokenizer.from_pretrained(str(cand["dir"]))
            _session = ort.InferenceSession(
                str(cand["file"]), providers=["CPUExecutionProvider"]
            )
            _active_model = cand["name"]
            logger.info(f"精排器已加载: {cand['name']} (ONNX, CPU)")
            return True

        logger.warning(
            "未找到精排模型(cache/bge-reranker-base 或 cache/ms-marco-onnx), "
            "精排降级为 RRF 结果"
        )
        return False
    except Exception as e:  # 任何加载失败都静默降级
        logger.warning(f"精排器加载失败, 降级为 RRF 结果: {e}")
        _session = None
        _tokenizer = None
        _active_model = None
        return False


def available() -> bool:
    """模型文件存在且能加载。"""
    return _load()


def active_model() -> Optional[str]:
    """当前使用的精排模型名(调试/日志用)。"""
    return _active_model if _load() else None


def score_pair(query: str, document: str) -> float:
    """单对 (query, document) 精排打分。调用方保证 available()。"""
    enc = _tokenizer(
        query, document, truncation=True, max_length=512, return_tensors="np"
    )
    feed = {k: v.astype("int64") for k, v in enc.items()}
    logits = _session.run(None, feed)[0]
    return float(logits[0][0])


def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """对候选 docs 精排, 返回按精排分降序的 docs(带 rerank_score)。

    不修改原 dict; 精排分写入副本的 rerank_score 字段。
    """
    if not docs or not _load():
        return docs
    scored: List[Dict[str, Any]] = []
    for doc in docs:
        content = doc.get("content") or ""
        if not content:
            continue
        try:
            s = score_pair(query, content)
        except Exception as e:
            logger.warning(f"精排打分失败, 跳过: {e}")
            continue
        item = dict(doc)
        item["rerank_score"] = s
        scored.append(item)
    scored.sort(key=lambda d: d["rerank_score"], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored
