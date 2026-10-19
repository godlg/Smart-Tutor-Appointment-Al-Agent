# services/knowledge_service.py

import hashlib
from dataclasses import dataclass, field

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from db.db_router import DatabaseRouter
from config.settings import settings
from . import document_parser
from .text_chunker import ChunkerSettings, TextChunker
from .text_embedding import embed_input
from observability import trace_span
import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# P1-13: 上传 DTO (frozen, immutable)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class UploadRequest:
    """上传一份 PDF / Markdown 的入参。
    frozen:上传流程中途任何模块都不能改 bytes / category,避免误覆盖。
    """
    file_bytes: bytes
    filename: str
    content_type: Optional[str]
    category: str
    keywords: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    doc_type: Optional[str] = None
    allow_duplicate: bool = False


@dataclass(frozen=True)
class UploadResult:
    """上传结果。
    skipped=True 时, doc_ids / chunk_count 反映已存在的入库结果。
    """
    doc_ids: List[int]
    source_hash: str
    chunk_count: int
    skipped: bool
    kind: str  # "pdf" | "markdown"


class KnowledgeService:
    """知识库服务类 - 结合数据库存储和向量检索"""

    def __init__(self, db_path: str = 'sqlite:///data/smart_appointment.db'):
        # 使用统一的DatabaseRouter，符合架构设计
        self.db_router = DatabaseRouter(db_path)
        # P1-9 修正: 统一使用 .knowledge_repo (消除双 alias)
        self.db = self.db_router.knowledge_repo
        self.index = None
        self.document_ids = []  # 维护文档ID与索引位置的映射
        self.initialized = False

        # 默认知识库内容（家教场景，P1-9）
        # 数据库已有 10 条记录, initialize() 检测到会跳过初始化.
        # 这里保留兜底集, 用于数据库被清空后的 fallback.
        # P1-12: 每条记录都打 subject / grade_level / doc_type 标签,
        # 让 search() 的 filter-then-vector 模式可以正常工作.
        self.default_knowledge = [
            {
                "content": "高中数学 12 周提分方案：函数、几何、数列、概率四大模块系统讲，重点章节以高考真题为载体，包含一对一作业批改、错题梳理、阶段测试三个环节，目标是高考数学从平均分提升 20-30 分。",
                "category": "course_intro",
                "keywords": ["高中", "数学", "高考", "12周", "提分", "课程介绍"],
                "subject": "数学",
                "grade_level": "高中",
                "doc_type": "course_intro",
            },
            {
                "content": "高三冲刺套餐：针对 6 月高考设计，1 月到 5 月完成三轮复习，包含考点串讲、模拟考刷题、心理疏导、报考志愿指导，适合数学、英语、理综等全学科。",
                "category": "course_intro",
                "keywords": ["高三", "冲刺", "高考", "复习", "套餐"],
                # 跨学科: subject 留 None(不应被特定学科过滤掉)
                "subject": None,
                "grade_level": "高中",
                "doc_type": "course_intro",
            },
            {
                "content": "小学阅读启蒙课：以绘本为载体，每节课讲解一本经典绘本 + 词汇 + 简单造句 + 复述故事，目标是让孩子爱上阅读，建立自主阅读习惯。",
                "category": "course_intro",
                "keywords": ["小学", "阅读", "绘本", "启蒙", "阅读习惯"],
                "subject": "语文",
                "grade_level": "小学",
                "doc_type": "course_intro",
            },
            {
                "content": "数学思维训练方法：1. 重视概念定义 2. 一题多解 3. 错题本 4. 主动总结题型。小学阶段最忌讳死记硬背公式，应该让孩子理解公式背后的几何直观。",
                "category": "subject_method",
                "keywords": ["数学", "思维", "方法", "小学", "错题本"],
                "subject": "数学",
                "grade_level": "小学",
                "doc_type": "subject_method",
            },
            {
                "content": "初中数学 一元二次方程 解题套路与错题梳理方法：重点讲解判别式、根与系数关系(韦达定理)、动点问题等高频考点，配中考真题训练。",
                "category": "subject_method",
                "keywords": ["初中", "数学", "一元二次方程", "韦达定理", "中考"],
                "subject": "数学",
                "grade_level": "初中",
                "doc_type": "subject_method",
            },
            {
                "content": "如何帮孩子选择合适的家教老师：先看学科匹配（数学 / 物理 / 英语等），再看年级段（小学 / 初中 / 高中），最后考虑风格（严格 / 温和 / 启发式）和孩子的接受度。建议先安排一次 30 分钟试讲再决定。",
                "category": "parent_guide",
                "keywords": ["家长", "选择", "老师", "试讲", "匹配"],
                # 跨学科+跨年级: subject / grade 都留 None
                "subject": None,
                "grade_level": None,
                "doc_type": "parent_guide",
            },
        ]

    async def initialize(self):
        """初始化知识库服务"""
        try:
            # 检查数据库中是否已有数据
            existing_docs = self.db.get_all_documents()

            if not existing_docs:
                logger.info("数据库为空，初始化默认知识库")
                await self._create_default_knowledge()
            else:
                logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")

            # 构建向量索引
            await self._build_vector_index()
            self.initialized = True
            logger.info("知识库服务初始化完成")

        except Exception as e:
            logger.error(f"知识库服务初始化失败: {e}")
            raise

    async def _create_default_knowledge(self):
        """创建默认知识库"""
        for knowledge in self.default_knowledge:
            try:
                # 生成嵌入向量
                text_for_embedding = f"{knowledge['content']} {' '.join(knowledge['keywords'])}"
                embedding = embed_input(text_for_embedding)

                # 保存到数据库(P1-12: 同步写元数据标签)
                self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                    embedding=embedding,
                    subject=knowledge.get('subject'),
                    grade_level=knowledge.get('grade_level'),
                    doc_type=knowledge.get('doc_type'),
                )
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")

            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    async def _build_vector_index(self):
        """构建向量索引"""
        try:
            documents = self.db.get_all_documents()
            if not documents:
                logger.warning("没有文档可用于构建索引")
                return

            embeddings = []
            self.document_ids = []

            for doc in documents:
                if doc.get('embedding'):
                    embeddings.append(doc['embedding'])
                    self.document_ids.append(doc['id'])
                else:
                    # 如果没有嵌入向量，生成一个
                    logger.warning(f"文档 {doc['id']} 缺少嵌入向量，正在生成...")
                    text_for_embedding = f"{doc['content']} {' '.join(doc.get('keywords', []))}"
                    embedding = embed_input(text_for_embedding)

                    # 更新数据库
                    self.db.update_document(doc['id'], embedding=embedding)

                    embeddings.append(embedding)
                    self.document_ids.append(doc['id'])

            # 防御:不同维度向量(测试残留 / 模型切换残留)会让 np.array 失败,
            # 用第一个非空 embedding 的 dim 作为基准,过滤掉其它维度的。
            if embeddings:
                ref_dim = len(embeddings[0])
                filtered_pairs = [
                    (e, did) for e, did in zip(embeddings, self.document_ids)
                    if len(e) == ref_dim
                ]
                skipped = len(embeddings) - len(filtered_pairs)
                if skipped:
                    logger.warning(
                        f"跳过 {skipped} 个维度不一致的 embedding "
                        f"(ref_dim={ref_dim})"
                    )
                embeddings = [e for e, _ in filtered_pairs]
                self.document_ids = [did for _, did in filtered_pairs]

            if embeddings:
                # 创建FAISS索引
                embeddings_array = np.array(embeddings).astype('float32')
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
                self.index.add(embeddings_array)
                logger.info(f"构建向量索引完成，包含 {len(embeddings)} 个向量")
            else:
                logger.warning("没有有效的嵌入向量，无法构建索引")

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 3,
        category: str = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict]:
        """搜索相关文档(薄包装: 等价于 search_multi([query]))。

        P1-12 范式 A (filter-then-vector):
        - 若传了 subject / grade_level / doc_type 任一,先用 SQL 缩候选集,
          再在子集内跑 FAISS 排序;
        - 全不传时,维持原行为(全集向量搜)。

        Args:
            query: 查询文本
            top_k: 返回 top_k 条
            category: 文档分类过滤 (兼容旧 API)
            subject: 学科硬约束, None 表示不筛选
            grade_level: 年级段硬约束, None 表示不筛选
            doc_type: 文档类型硬约束, None 表示不筛选

        Returns:
            按相似度降序的文档列表;空列表表示无命中
        """
        return await self.search_multi(
            [query], top_k=top_k, category=category,
            subject=subject, grade_level=grade_level, doc_type=doc_type,
        )

    async def search_multi(
        self,
        queries: List[str],
        top_k: int = 3,
        category: str = None,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict]:
        """多查询 Dense 检索: 每条 query 各检索一次, 按 doc_id 做 RRF 融合后截 top_k。

        单条 query 时逐字沿用原检索路径(分数 = FAISS 相似度),不引入任何变化;
        多条 query 时每条各取 rrf_candidates 深度,RRF 融合后取 top_k,
        score 记为 rrf_score(跨 query 的分数量纲不可比,融合分才是可比的)。
        """
        if not queries:
            queries = [""]
        if not self.initialized or self.index is None:
            logger.warning("知识库服务未初始化或索引不可用")
            return []

        # 若被增删后索引被标脏,先 lazy rebuild 再搜
        await self._ensure_index_fresh()

        has_meta_filter = any(
            v is not None for v in (subject, grade_level, doc_type)
        )
        per_query_k = top_k if len(queries) == 1 else max(
            top_k, settings.rrf_candidates
        )

        # Trace 埋点(dense 路径):包一层 rag_search span,与 hybrid 同链路可嵌套
        with trace_span(
            "rag_search",
            mode="dense",
            query=queries[0][:30],
            top_k=top_k,
            has_meta_filter=has_meta_filter,
        ) as span:
            per_query_docs: List[List[Dict]] = []
            for query in queries:
                if not has_meta_filter:
                    docs = await self._vector_search_full(
                        query=query, top_k=per_query_k, category=category,
                    )
                else:
                    docs = await self._vector_search_with_metadata_filter(
                        query=query, top_k=per_query_k, category=category,
                        subject=subject, grade_level=grade_level, doc_type=doc_type,
                    )
                per_query_docs.append(docs)

            if len(queries) == 1:
                results = per_query_docs[0]
            else:
                results = self._fuse_dense(per_query_docs, top_k)
            span.attrs["result_count"] = len(results)
            span.attrs["variants"] = len(queries)
            return results

    @staticmethod
    def _fuse_dense(
        per_query_docs: List[List[Dict]], top_k: int
    ) -> List[Dict]:
        """按 doc_id 做 RRF 融合, 截 top_k;同一文档只保留一条(排名靠前的那份)。"""
        from services.hybrid_retriever import rrf_merge

        ranked_lists = [
            [(int(d["id"]), rank) for rank, d in enumerate(docs, start=1)]
            for docs in per_query_docs
        ]
        first_seen: Dict[int, Dict] = {}
        for docs in per_query_docs:
            for doc in docs:
                first_seen.setdefault(int(doc["id"]), doc)

        fused: List[Dict] = []
        for doc_id, rrf_score in rrf_merge(ranked_lists, settings.rrf_k)[:top_k]:
            doc = first_seen.get(doc_id)
            if doc is None:
                continue
            item = dict(doc)
            item["score"] = rrf_score
            item["rank"] = len(fused) + 1
            fused.append(item)
        return fused

    async def _vector_search_full(
        self, query: str, top_k: int, category: Optional[str],
    ) -> List[Dict]:
        """原行为:全集向量搜,category 过滤发生在结果集上。"""
        try:
            query_embedding = embed_input(query)
            query_array = np.array([query_embedding]).astype('float32')

            scores, indices = self.index.search(
                query_array, min(top_k * 2, len(self.document_ids))
            )

            results: List[Dict] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    doc = self.db.get_document(doc_id)

                    if doc:
                        if category and doc.get('category') != category:
                            continue

                        doc['score'] = float(score)
                        doc['rank'] = len(results) + 1
                        results.append(doc)

                        if len(results) >= top_k:
                            break

            return results

        except Exception as e:
            logger.error(f"搜索知识库失败: {e}", exc_info=True)
            return []

    async def _vector_search_with_metadata_filter(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        subject: Optional[str],
        grade_level: Optional[str],
        doc_type: Optional[str],
    ) -> List[Dict]:
        """filter-then-vector:先用 SQL 缩候选集,再在子集上跑 FAISS。

        设计要点:
        1. SQL 过滤在向量搜索之前: 即便"高一数学"和"高三数学"向量很近,
           grade_level=初中 也能硬把高一/高三全部挡掉。
        2. 子索引每次按需构造: 候选集通常 < 50 条, 直接新建 IndexFlatIP 即可,
           不污染 self.index 全集索引。
        """
        try:
            # 1. SQL 过滤(业务硬约束)
            candidates = self.db.search_documents_with_filters(
                subject=subject,
                grade_level=grade_level,
                doc_type=doc_type,
            )
            if not candidates:
                return []

            # 2. 子集构建索引
            sub_embeddings = []
            sub_doc_ids: List[int] = []
            for doc in candidates:
                if doc.get('embedding'):
                    sub_embeddings.append(doc['embedding'])
                    sub_doc_ids.append(doc['id'])

            if not sub_embeddings:
                return []

            sub_array = np.array(sub_embeddings).astype('float32')
            dimension = sub_array.shape[1]
            sub_index = faiss.IndexFlatIP(dimension)
            sub_index.add(sub_array)

            # 3. 在子集上搜
            query_embedding = embed_input(query)
            query_array = np.array([query_embedding]).astype('float32')
            k = min(top_k, len(sub_doc_ids))
            scores, indices = sub_index.search(query_array, k)

            results: List[Dict] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(sub_doc_ids):
                    continue
                doc_id = sub_doc_ids[idx]
                doc = self.db.get_document(doc_id)
                if not doc:
                    continue
                # 旧 category 过滤(若调用方同时传了 category,仍生效)
                if category and doc.get('category') != category:
                    continue
                doc['score'] = float(score)
                doc['rank'] = len(results) + 1
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"带元数据过滤的搜索失败: {e}", exc_info=True)
            return []

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档 (增量,不重建索引)"""
        try:
            if keywords is None:
                keywords = []

            # 生成嵌入向量
            text_for_embedding = f"{content} {' '.join(keywords)}"
            embedding = embed_input(text_for_embedding)

            # 保存到数据库
            doc_id = self.db.add_document(content, category, keywords, embedding)

            # 增量加入索引(不重建)
            await self._add_to_index(doc_id, embedding)

            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(self, doc_id: int, content: str = None, category: str = None, keywords: List[str] = None) -> bool:
        """更新文档 (删旧向量,加新向量)"""
        try:
            # 如果更新了内容或关键词，需要重新生成嵌入向量
            embedding = None
            if content is not None or keywords is not None:
                # 获取当前文档信息
                current_doc = self.db.get_document(doc_id)
                if not current_doc:
                    return False

                # 使用新值或保持原值
                final_content = content if content is not None else current_doc['content']
                final_keywords = keywords if keywords is not None else current_doc.get('keywords', [])

                # 生成新的嵌入向量
                text_for_embedding = f"{final_content} {' '.join(final_keywords)}"
                embedding = embed_input(text_for_embedding)

            # 更新数据库
            success = self.db.update_document(doc_id, content, category, keywords, embedding)

            if success:
                # 更新向量(单条:删旧,加新),避免全文重建
                if embedding is not None:
                    await self._replace_in_index(doc_id, embedding)
                else:
                    # 只改 category/keywords(不动 embedding),索引里向量不变
                    pass

            return success

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """删除文档 (从索引中移除单条向量)"""
        try:
            success = self.db.delete_document(doc_id, soft_delete)

            if success:
                # 从索引中移除(不重建)
                self._remove_from_index(doc_id)

            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    # ------------------- 增量索引操作 -------------------

    async def _add_to_index(self, doc_id: int, embedding) -> None:
        """增量:把单个向量加到 FAISS 索引"""
        if self.index is None:
            await self._build_vector_index()
            return
        import numpy as np
        vec = np.array([embedding]).astype('float32')
        self.index.add(vec)
        self.document_ids.append(doc_id)

    def _remove_from_index(self, doc_id: int) -> bool:
        """增量:从索引中移除某个 doc_id 对应的向量。
        FAISS 不支持原地删除单条,只能重建;但只在 document_ids 里去掉引用。
        之后调用 _build_vector_index() 时会得到干净索引。
        """
        if doc_id in self.document_ids:
            self.document_ids.remove(doc_id)
            # 把 index 标记为需要重建 (下次 search 或显式 rebuild 时重建)
            self._index_dirty = True
            return True
        return False

    async def _replace_in_index(self, doc_id: int, new_embedding) -> None:
        """增量:把某个 doc_id 的向量替换成新向量。"""
        if self.index is None:
            await self._build_vector_index()
            return
        # FAISS 不能原地替换,最干净的策略:标记脏 + 下次 lazy rebuild
        self._remove_from_index(doc_id)
        await self._add_to_index(doc_id, new_embedding)

    async def rebuild_index(self) -> None:
        """显式触发全量重建 (大批量修改后调用一次)。"""
        self._index_dirty = True
        await self._build_vector_index()

    async def _ensure_index_fresh(self) -> None:
        """若索引被标脏,先重建再返回。"""
        if getattr(self, "_index_dirty", False):
            await self._build_vector_index()
            self._index_dirty = False

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict]:
        """获取所有文档"""
        return self.db.get_all_documents(include_inactive)

    def get_document(self, doc_id: int) -> Dict:
        """获取指定文档"""
        return self.db.get_document(doc_id)

    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        return self.db.get_all_categories()

    def get_documents_count(self) -> int:
        """获取文档总数"""
        return self.db.get_documents_count()

    def search_by_category(self, category: str) -> List[Dict]:
        """按分类搜索文档"""
        return self.db.search_documents_by_category(category)

    def search_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """按关键词搜索文档"""
        return self.db.search_documents_by_keywords(keywords)

    # ------------------------------------------------------------------
    # P1-13: 文件上传入口
    # ------------------------------------------------------------------

    async def add_uploaded_document(self, req: UploadRequest) -> UploadResult:
        """把 PDF / Markdown 文件解析 → 分块 → 嵌入 → 入库 → 增量进 FAISS。

        流程:
        1. sha256(file_bytes) 取 source_hash
        2. 若 not allow_duplicate 且 hash 已存在 → 返回 skipped
        3. document_parser.parse → text + title + kind
        4. TextChunker(settings.chunk_size/overlap).split → List[TextChunk]
        5. 每 chunk:embed → repo.add_document(...溯源列) → _add_to_index

        Raises:
            document_parser.UnsupportedFileType / EmptyDocument:
                让 API 层映射到 400
        """
        # 1. SHA256 去重键
        source_hash = hashlib.sha256(req.file_bytes).hexdigest()

        # 2. 去重检查
        if not req.allow_duplicate:
            existing = self.db.find_chunks_by_source_hash(source_hash)
            if existing:
                logger.info(
                    f"上传去重命中:source_hash={source_hash[:8]}..., "
                    f"已存在 {len(existing)} 个 chunk,跳过入库"
                )
                return UploadResult(
                    doc_ids=[r["id"] for r in existing],
                    source_hash=source_hash,
                    chunk_count=len(existing),
                    skipped=True,
                    kind="markdown",  # 此处 kind 不重要;前端不需要
                )

        # 3. 解析
        kind, parsed = document_parser.parse(
            req.file_bytes, req.filename, req.content_type,
        )

        # 4. 分块 (参数从 settings 读,与 config/components.yaml 的 splitter 声明对齐)
        chunks = TextChunker(
            ChunkerSettings(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        ).split(parsed.text)
        if not chunks:
            # splitter 偶尔返回空列表(全是空白),防御性抛错
            raise document_parser.EmptyDocument("分块结果为空")

        # 5. 逐 chunk 入库 + 增量索引
        doc_ids: List[int] = []
        for i, chunk in enumerate(chunks):
            embedding = embed_input(chunk.text)
            doc_id = self.db.add_document(
                content=chunk.text,
                category=req.category,
                keywords=req.keywords,
                embedding=embedding,
                subject=req.subject,
                grade_level=req.grade_level,
                doc_type=req.doc_type,
                source_file=req.filename,
                chunk_index=i,
                source_hash=source_hash,
                chunk_count=len(chunks),
            )
            await self._add_to_index(doc_id, embedding)
            doc_ids.append(doc_id)

        logger.info(
            f"上传入库成功:file={req.filename!r} kind={kind} "
            f"chunks={len(chunks)} doc_ids={doc_ids}"
        )

        return UploadResult(
            doc_ids=doc_ids,
            source_hash=source_hash,
            chunk_count=len(chunks),
            skipped=False,
            kind=kind,
        )