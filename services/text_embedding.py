"""
文本 Embedding 工具（家教场景）

为知识库 / 老师匹配提供 embedding 计算与相似度匹配。
底层通过 config.model_provider.create_embedding_model() 注入，
默认走智谱 embedding-3（兼容 OpenAI 接口）。
"""

import numpy as np
import faiss
from typing import List

from config.model_provider import create_embedding_model


def embed_input(input_text: str) -> List[float]:
    """计算一段文本的 embedding 向量"""
    model = create_embedding_model()
    return model.embed_query(input_text)


def find_best_match_indices(text: str, candidates: list) -> list:
    """
    输入一个 text 和一个候选列表，通过 embedding + FAISS 索引返回所有候选项的索引，按相似度从高到低排序
    """
    if not candidates:
        return []
    candidate_embs = [embed_input(c) for c in candidates]
    candidate_embs = np.array(candidate_embs).astype("float32")
    dimension = candidate_embs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(candidate_embs)
    text_emb = np.array([embed_input(text)]).astype("float32")
    k = len(candidates)
    _, indices = index.search(text_emb, k)
    return indices[0][:k].tolist()
