"""一次性重建知识库向量索引 (doubao 2048 维)。

背景: .env 已切换 EMBEDDING_PROVIDER=doubao (dim=2048),
而 knowledge_documents 表的存量 embedding 是旧模型 1024 维, 导致 FAISS
建索引 ref_dim=1024、search 时 2048 维 query 触发 AssertionError, 咨询检索
异常落「教育学常识」兜底。

本脚本仅重写 knowledge_documents.embedding 列:
  - 用 services.text_embedding.embed_input (doubao, 2048 维) 对
    content + keywords 重新生成向量;
  - 通过仓库层 repo.update_document(doc_id, embedding=...) 写回, 保持
    JSON 序列化格式与业务逻辑一致 (勿裸 SQL);
  - 幂等: 重复运行覆盖写, 安全。
  - 不改 content / category / keywords / 任何元数据, 不删行, 不清空其它表。

用法:
  ./.venv/Scripts/python.exe scripts/rebuild_knowledge_embeddings.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# 允许从任意 cwd 直接运行: 把项目根加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.knowledge_service import KnowledgeService
from services.text_embedding import embed_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def rebuild() -> None:
    svc = KnowledgeService()
    documents = svc.get_all_documents(include_inactive=False)
    total = len(documents)
    print(f"[rebuild] 待重建文档数: {total}")

    ok, failed, dims = 0, 0, set()
    for i, doc in enumerate(documents, start=1):
        doc_id = doc["id"]
        content = doc.get("content") or ""
        keywords = doc.get("keywords") or []
        text_for_embedding = f"{content} {' '.join(keywords)}"

        # 生成新向量 (doubao, 2048 维)
        embedding = embed_input(text_for_embedding)
        dims.add(len(embedding))
        print(f"[{i}/{total}] doc_id={doc_id} dim={len(embedding)}")

        # 写回 (仅 embedding 列)
        updated = svc.db.update_document(doc_id, embedding=embedding)
        if not updated:
            print(f"[{i}/{total}] doc_id={doc_id} 更新失败!")
            failed += 1
            continue
        ok += 1

    print(f"[rebuild] 完成: 成功 {ok}, 失败 {failed}, 维度分布 {sorted(dims)}")
    if failed:
        raise SystemExit(f"[rebuild] 有 {failed} 条更新失败, 请检查")
    if dims != {2048}:
        raise SystemExit(f"[rebuild] 维度异常, 预期全部 2048, 实际 {sorted(dims)}")


if __name__ == "__main__":
    asyncio.run(rebuild())