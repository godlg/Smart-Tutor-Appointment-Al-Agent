"""
知识检索器

负责从知识库中检索相关信息
"""

import logging
from typing import List, Dict, Any, Optional
from services.knowledge_service import KnowledgeService
from services.hybrid_retriever import HybridRetriever
from config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """知识检索器"""

    def __init__(self):
        self.knowledge_service = KnowledgeService()
        self.kb_initialized = False
        # 混合检索器: 与 Dense 主路径解耦, RETRIEVAL_MODE=hybrid 时启用
        self.hybrid_retriever = HybridRetriever(
            knowledge_service=self.knowledge_service,
            rrf_k=settings.rrf_k,
            candidates=settings.rrf_candidates,
            rerank=settings.rerank_enabled,
        )
        self.retrieval_mode = settings.retrieval_mode

    async def initialize(self):
        """初始化知识库服务。

        降级：知识库不可用（表缺失 / DB 异常 / 连接失败）时不抛错，仅记录日志并保持
        `kb_initialized=False`（下一次调用会重试，DB 恢复后可自愈），让咨询走"无知识库"
        的既有降级路径，而不是把整轮对话炸掉。
        """
        if not self.kb_initialized:
            try:
                await self.knowledge_service.initialize()
                self.kb_initialized = True
                print("✅ 咨询机器人知识库服务已初始化")
            except Exception as exc:  # noqa: BLE001 - 知识库不可用不应中断对话
                logger.warning("知识库初始化失败，降级为无知识库继续: %s", exc)

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 3,
        query_context: str = "",
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """搜索相关知识。

        query_context: 来自 ConversationContext 的历史摘要文本,
        拼到 query 前面一起送进 FAISS 检索(不传到知识库 metadata,只做语义召回增强)。
        为空时保持原行为,仅用单条输入查询。

        subject / grade_level (P0-1): 从会话上下文抽出的学科 / 年级段硬约束,
        透传给底层 filter-then-vector 检索(SQL 先缩候选集,再在子集内做向量排序)。
        两者都为 None 时行为与改动前完全一致(全量向量检索,向后兼容)。

        降级：检索过程抛错（表缺失 / DB 异常 / 向量计算失败）时返回空列表，
        由调用方走"未找到相关知识"的既有降级文案；**不**把异常抛给上层炸掉整轮。
        """
        # 确保知识库已初始化
        if not self.kb_initialized:
            await self.initialize()

        # 历史上下文拼到查询前,让"计算薄弱"这类指代不清的追问能借历史(初二、数学)召回更准
        effective_query = (query_context + "\n" + query).strip() if query_context else query

        # 搜索相关知识(实际送入检索的是 effective_query)
        try:
            if self.retrieval_mode == "hybrid":
                relevant_docs = await self.hybrid_retriever.search(
                    effective_query, top_k=top_k,
                    subject=subject, grade_level=grade_level,
                )
            else:
                relevant_docs = await self.knowledge_service.search(
                    effective_query, top_k=top_k,
                    subject=subject, grade_level=grade_level,
                )
        except Exception as exc:  # noqa: BLE001 - 检索失败降级为空召回
            logger.warning("知识库检索失败，降级为空召回: %s", exc)
            relevant_docs = []

        # 记录检索日志
        self._log_search_results(
            effective_query, relevant_docs,
            subject=subject, grade_level=grade_level,
        )

        return relevant_docs or []

    def _log_search_results(
        self,
        query: str,
        relevant_docs: List[Dict[str, Any]],
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
    ):
        """记录搜索结果日志(带本轮生效的元数据硬约束,便于端到端排查召回)。"""
        filters = f" [学科:{subject or '-'} 年级:{grade_level or '-'}]"
        if relevant_docs:
            print(f"🔍 知识库检索结果{filters} (查询: '{query}'):")
            for i, doc in enumerate(relevant_docs, 1):
                score = doc.get('score', 0)
                category = doc.get('category', '未知')
                content = doc.get('content', '')[:80]
                print(f"  {i}. [相关度:{score:.3f}] [分类:{category}] {content}...")
            print(f"📊 知识库统计: 共检索到 {len(relevant_docs)} 条相关知识")
        else:
            print(f"⚠️ 知识库检索{filters}: 未找到与 '{query}' 相关的知识")
