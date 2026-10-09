from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import or_
from ..base.interfaces import BaseKnowledgeRepository
from ..base.session_manager import SessionManager
from ..models import KnowledgeDocument


class KnowledgeRepository(BaseKnowledgeRepository):
    """
    知识库数据访问对象
    
    职责：
    1. 知识文档的CRUD操作
    2. 文档搜索和分类
    3. 文档状态管理
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        初始化知识库数据仓库
        
        Args:
            session_manager: 会话管理器
        """
        self.session_manager = session_manager

    def add_document(self, content: str, category: str, keywords: Optional[List[str]] = None,
                    embedding: Optional[List[float]] = None,
                    subject: Optional[str] = None,
                    grade_level: Optional[str] = None,
                    doc_type: Optional[str] = None,
                    source_file: Optional[str] = None,
                    chunk_index: Optional[int] = None,
                    source_hash: Optional[str] = None,
                    chunk_count: Optional[int] = None) -> int:
        """
        添加知识文档

        Args:
            content: 文档内容
            category: 文档分类
            keywords: 关键词列表
            embedding: 嵌入向量
            subject: 学科标签 (SubjectEnum 值),可选,用于 hybrid 检索前置硬筛选
            grade_level: 年级段标签 (GradeLevelEnum 值),可选
            doc_type: 文档类型标签 (KnowledgeCategoryEnum 值),可选
            source_file: (P1-13) 上传溯源,原始文件名
            chunk_index: (P1-13) 文件内 0-based 位置
            source_hash: (P1-13) 文件 SHA256,用于去重
            chunk_count: (P1-13) 文件总块数

        Returns:
            新创建的文档ID
        """
        with self.session_manager.session_scope() as session:
            document = KnowledgeDocument(
                content=content,
                category=category,
                keywords=keywords,
                embedding=embedding,
                subject=subject,
                grade_level=grade_level,
                doc_type=doc_type,
                source_file=source_file,
                chunk_index=chunk_index,
                source_hash=source_hash,
                chunk_count=chunk_count,
            )
            session.add(document)
            session.flush()
            return document.id

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档信息字典，如果不存在返回None
        """
        with self.session_manager.session_scope() as session:
            document = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.id == doc_id
            ).first()
            
            if not document:
                return None
                
            return self._document_to_dict(document)

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        获取所有文档
        
        Args:
            include_inactive: 是否包含已删除的文档
            
        Returns:
            文档信息列表
        """
        with self.session_manager.session_scope() as session:
            query = session.query(KnowledgeDocument)
            
            if not include_inactive:
                query = query.filter(KnowledgeDocument.is_active == 1)
                
            documents = query.all()
            return [self._document_to_dict(doc) for doc in documents]

    def update_document(self, doc_id: int, content: Optional[str] = None, category: Optional[str] = None,
                       keywords: Optional[List[str]] = None, embedding: Optional[List[float]] = None,
                       subject: Optional[str] = None, grade_level: Optional[str] = None,
                       doc_type: Optional[str] = None) -> bool:
        """
        更新文档

        Args:
            doc_id: 文档ID
            content: 新内容
            category: 新分类
            keywords: 新关键词
            embedding: 新嵌入向量
            subject: 新学科标签
            grade_level: 新年级段标签
            doc_type: 新文档类型标签

        Returns:
            更新是否成功
        """
        with self.session_manager.session_scope() as session:
            document = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.id == doc_id
            ).first()

            if not document:
                return False

            if content is not None:
                document.content = content
            if category is not None:
                document.category = category
            if keywords is not None:
                document.keywords = keywords
            if embedding is not None:
                document.embedding = embedding
            if subject is not None:
                document.subject = subject
            if grade_level is not None:
                document.grade_level = grade_level
            if doc_type is not None:
                document.doc_type = doc_type

            document.updated_at = datetime.utcnow()
            return True

    def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """
        删除文档（支持软删除）
        
        Args:
            doc_id: 文档ID
            soft_delete: 是否软删除
            
        Returns:
            删除是否成功
        """
        with self.session_manager.session_scope() as session:
            document = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.id == doc_id
            ).first()
            
            if not document:
                return False
            
            if soft_delete:
                document.is_active = 0
                document.updated_at = datetime.utcnow()
            else:
                session.delete(document)
            
            return True

    def search_documents_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        按分类搜索文档
        
        Args:
            category: 文档分类
            
        Returns:
            匹配的文档列表
        """
        with self.session_manager.session_scope() as session:
            documents = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.category == category,
                KnowledgeDocument.is_active == 1
            ).all()
            
            return [self._document_to_dict(doc) for doc in documents]

    def search_documents_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        按关键词搜索文档
        
        Args:
            keywords: 关键词列表
            
        Returns:
            匹配的文档列表
        """
        with self.session_manager.session_scope() as session:
            documents = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.is_active == 1
            ).all()
            
            # 简单的关键词匹配
            matched_docs = []
            for doc in documents:
                doc_keywords = doc.keywords or []
                if any(keyword in doc_keywords for keyword in keywords):
                    matched_docs.append(self._document_to_dict(doc))
            
            return matched_docs

    def search_documents_by_content(self, search_text: str) -> List[Dict[str, Any]]:
        """
        按内容搜索文档

        Args:
            search_text: 搜索文本

        Returns:
            匹配的文档列表
        """
        with self.session_manager.session_scope() as session:
            documents = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.content.contains(search_text),
                KnowledgeDocument.is_active == 1
            ).all()

            return [self._document_to_dict(doc) for doc in documents]

    def search_documents_with_filters(
        self,
        subject: Optional[str] = None,
        grade_level: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """元数据过滤查询:用 SQL WHERE 硬匹配缩候选集。

        P1-12: 学科 / 年级 / 文档类型是家教业务的硬约束,
        必须在向量检索之前用 SQL 锁死,否则召回会被语义近邻污染。

        所有过滤参数都允许 None(不筛选);多条过滤条件 AND 组合。
        不包含已软删除(is_active=0)的文档。

        P0-2 NULL 豁免:subject / grade_level 为 NULL 表示「跨学科 / 跨年级」,
        这类文档不应被任何具体学科/年级的过滤误杀(默认知识库里的「高三冲刺套餐」
        subject=None、「如何选择家教老师」subject/grade 都 None 都靠这条语义存活)。
        因此 subject / grade_level 用 `== 值 OR IS NULL` 匹配;
        doc_type 保持严格等值(跨类型文档没有豁免语义)。
        """
        with self.session_manager.session_scope() as session:
            query = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.is_active == 1
            )
            if subject is not None:
                query = query.filter(
                    or_(
                        KnowledgeDocument.subject == subject,
                        KnowledgeDocument.subject.is_(None),
                    )
                )
            if grade_level is not None:
                query = query.filter(
                    or_(
                        KnowledgeDocument.grade_level == grade_level,
                        KnowledgeDocument.grade_level.is_(None),
                    )
                )
            if doc_type is not None:
                query = query.filter(KnowledgeDocument.doc_type == doc_type)
            documents = query.all()
            return [self._document_to_dict(doc) for doc in documents]

    def get_all_categories(self) -> List[str]:
        """
        获取所有分类
        
        Returns:
            分类列表
        """
        with self.session_manager.session_scope() as session:
            categories = session.query(KnowledgeDocument.category).filter(
                KnowledgeDocument.is_active == 1
            ).distinct().all()
            
            return [cat[0] for cat in categories]

    def get_documents_count(self) -> int:
        """
        获取文档总数
        Returns:
            活跃文档数量
        """
        with self.session_manager.session_scope() as session:
            return session.query(KnowledgeDocument).filter(
                KnowledgeDocument.is_active == 1
            ).count()

    def get_documents_by_category_count(self) -> Dict[str, int]:
        """
        获取各分类的文档数量
        
        Returns:
            分类和文档数量的字典
        """
        with self.session_manager.session_scope() as session:
            from sqlalchemy import func
            
            result = session.query(
                KnowledgeDocument.category,
                func.count(KnowledgeDocument.id).label('count')
            ).filter(
                KnowledgeDocument.is_active == 1
            ).group_by(KnowledgeDocument.category).all()
            
            return {category: count for category, count in result}

    def _document_to_dict(self, document: KnowledgeDocument) -> Dict[str, Any]:
        """将文档对象转换为字典"""
        return {
            'id': document.id,
            'content': document.content,
            'category': document.category,
            'keywords': document.keywords,
            'embedding': document.embedding,
            'subject': document.subject,
            'grade_level': document.grade_level,
            'doc_type': document.doc_type,
            'source_file': document.source_file,
            'chunk_index': document.chunk_index,
            'source_hash': document.source_hash,
            'chunk_count': document.chunk_count,
            'created_at': document.created_at,
            'updated_at': document.updated_at,
            'is_active': bool(document.is_active)
        }

    def find_chunks_by_source_hash(self, source_hash: str) -> List[Dict[str, Any]]:
        """按 source_hash 查所有 chunk,按 chunk_index ASC 排序,排除软删除。

        P1-13:用于上传去重。空 hash 时返回空列表 (避免误命中所有 NULL 行)。
        """
        if not source_hash:
            return []
        with self.session_manager.session_scope() as session:
            rows = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.source_hash == source_hash,
                KnowledgeDocument.is_active == 1,
            ).order_by(KnowledgeDocument.chunk_index.asc()).all()
            return [self._document_to_dict(r) for r in rows]
    
