"""
知识库管理API
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/knowledge", tags=["知识库管理"])

# P1-13: 上传文件大小上限(10MB),超过抛 413
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _str_to_optional(s: Optional[str]) -> Optional[str]:
    """multipart Form 字段空串 → None(用于 subject/grade_level/doc_type 元数据)。"""
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None


class KnowledgeItem(BaseModel):
    id: int = None
    question: str
    answer: str
    category: str = "general"


class SearchRequest(BaseModel):
    query: str
    # P1-12: 元数据过滤参数(可选,None 表示不筛选)
    subject: str = None
    grade_level: str = None
    doc_type: str = None


@router.get("/")
async def get_all_knowledge():
    """获取所有知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entries = knowledge_service.get_all_documents()
        
        # 安全获取categories，避免出错
        try:
            categories = knowledge_service.get_all_categories()
        except Exception as e:
            print(f"获取categories失败: {e}")
            categories = []
        
        return {
            "documents": entries or [],
            "categories": categories or [],
            "total_count": len(entries) if entries else 0,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库失败: {str(e)}")


@router.post("/upload")
async def upload_knowledge(
    file: UploadFile = File(..., description="PDF 或 Markdown 文件,最大 10MB"),
    category: str = Form(..., description="文档分类,例:faq / subject_method"),
    keywords: str = Form("", description="逗号分隔的关键词,可空"),
    subject: Optional[str] = Form(None),
    grade_level: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    allow_duplicate: bool = Form(False),
):
    """通过 multipart 上传 PDF / Markdown,自动分块 + 嵌入 + 入库。

    Returns:
        200 OK: {status, data: {doc_ids, source_hash, chunk_count, skipped, kind}}
        400: 扩展名不支持 / 解析后为空
        413: 文件超过 10MB
        422: markitdown 解析失败(损坏 PDF / 加密 PDF / OCR 缺失)
        500: DB / 嵌入服务异常
    """
    # 补4: 业务级 metric (outcome 不依赖 HTTP status, 更准确反映业务结果)
    from observability import (
        KNOWLEDGE_UPLOAD_TOTAL,
        KNOWLEDGE_UPLOAD_CHUNKS,
        KNOWLEDGE_UPLOAD_BYTES,
    )

    try:
        file_bytes = await file.read()
    except Exception as e:
        KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="rejected_error", kind="unknown").inc()
        raise HTTPException(status_code=400, detail=f"读取上传文件失败: {e}")

    # 大小限制(必须在解析之前挡)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="rejected_too_large", kind="unknown").inc()
        raise HTTPException(
            status_code=413,
            detail=f"文件过大({len(file_bytes)} bytes),上限 {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        from services.knowledge_service import (
            KnowledgeService,
            UploadRequest,
        )
        from services.document_parser import (
            UnsupportedFileType,
            EmptyDocument,
            DocumentParseError,
            detect_kind,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务依赖加载失败: {e}")

    # 提前 detect_kind 用于 metric label (失败也不阻塞主流程)
    try:
        detected_kind = detect_kind(file.filename or "", file.content_type)
    except UnsupportedFileType:
        detected_kind = "unknown"

    knowledge_service = KnowledgeService()
    if not knowledge_service.initialized:
        await knowledge_service.initialize()

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    req = UploadRequest(
        file_bytes=file_bytes,
        filename=file.filename or "upload.bin",
        content_type=file.content_type,
        category=category,
        keywords=kw_list,
        subject=_str_to_optional(subject),
        grade_level=_str_to_optional(grade_level),
        doc_type=_str_to_optional(doc_type),
        allow_duplicate=allow_duplicate,
    )

    # bytes counter (按 detected_kind 统计,unknown 也计,反映真实流量)
    KNOWLEDGE_UPLOAD_BYTES.labels(kind=detected_kind).inc(len(file_bytes))

    try:
        result = await knowledge_service.add_uploaded_document(req)
    except UnsupportedFileType as e:
        KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="rejected_unsupported", kind=detected_kind,
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyDocument as e:
        KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="rejected_error", kind=detected_kind,
        ).inc()
        raise HTTPException(status_code=400, detail=f"解析后内容为空: {e}")
    except DocumentParseError as e:
        KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="rejected_error", kind=detected_kind,
        ).inc()
        raise HTTPException(status_code=422, detail=f"PDF 解析失败: {e}")
    except Exception as e:
        KNOWLEDGE_UPLOAD_TOTAL.labels(
            outcome="rejected_error", kind=detected_kind,
        ).inc()
        logger_msg = str(e)
        logger_module = __import__("logging").getLogger(__name__)
        logger_module.exception(f"上传入库失败: {logger_msg}")
        raise HTTPException(status_code=500, detail=f"上传入库失败: {e}")

    # 成功 / 去重命中 都是 outcome 终态
    if result.skipped:
        KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="skipped", kind=result.kind).inc()
    else:
        KNOWLEDGE_UPLOAD_TOTAL.labels(outcome="success", kind=result.kind).inc()
        # 只在非 dedup 跳过时计入 chunks (dedup 没有新内容入索引)
        KNOWLEDGE_UPLOAD_CHUNKS.labels(kind=result.kind).inc(result.chunk_count)

    return {
        "status": "success",
        "message": "文件上传成功" if not result.skipped else "文件已存在,已跳过(去重命中)",
        "data": {
            "doc_ids": result.doc_ids,
            "source_hash": result.source_hash,
            "chunk_count": result.chunk_count,
            "skipped": result.skipped,
            "kind": result.kind,
        },
    }


@router.get("/{knowledge_id}")
async def get_knowledge(knowledge_id: int):
    """获取特定知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entry = knowledge_service.get_document(knowledge_id)
        if not entry:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "data": entry
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识条目失败: {str(e)}")


@router.post("/")
async def add_knowledge(item: KnowledgeItem):
    """添加新的知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        content = f"问题: {item.question}\n答案: {item.answer}"
        result = await knowledge_service.add_document(
            content=content,
            category=item.category
        )
        return {
            "status": "success",
            "message": "知识条目添加成功",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加知识条目失败: {str(e)}")


@router.put("/{knowledge_id}")
async def update_knowledge(knowledge_id: int, item: KnowledgeItem):
    """更新知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        # 将问答组合成文档内容
        content = f"问题: {item.question}\n答案: {item.answer}"
        result = await knowledge_service.update_document(
            doc_id=knowledge_id,
            content=content,
            category=item.category
        )
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目更新成功",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新知识条目失败: {str(e)}")


@router.delete("/{knowledge_id}")
async def delete_knowledge(knowledge_id: int):
    """删除知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.delete_document(knowledge_id)
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除知识条目失败: {str(e)}")


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """搜索知识库

    P1-12: 支持元数据硬过滤(subject / grade_level / doc_type)。
    业务硬约束会先于向量搜索执行,避免语义近邻污染召回。
    """
    # 补4: 搜索调用次数 + 是否带元数据过滤
    from observability import KNOWLEDGE_SEARCH_TOTAL

    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        results = await knowledge_service.search(
            query=request.query,
            subject=request.subject,
            grade_level=request.grade_level,
            doc_type=request.doc_type,
        )

        has_meta = "true" if any(
            v is not None for v in (
                request.subject, request.grade_level, request.doc_type,
            )
        ) else "false"
        if not results:
            bucket = "none"
        elif len(results) <= 3:
            bucket = "few"
        else:
            bucket = "many"
        KNOWLEDGE_SEARCH_TOTAL.labels(
            has_meta_filter=has_meta, result_count_bucket=bucket,
        ).inc()

        return {
            "status": "success",
            "data": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索知识库失败: {str(e)}")
