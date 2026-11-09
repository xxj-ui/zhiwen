"""M1 文档管理路由：04 §4.1~§4.3"""
from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from server.api.deps import get_container
from server.core.errors import AppError
from server.api.schemas import DocumentChunks, DocumentList, Envelope, UploadItem, UploadResult
from server.services import documents

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=Envelope[UploadResult], summary="上传文档（每次 1~20 个）")
async def upload_document(files: list[UploadFile] = File(..., description="1~20 个文件，按提交顺序串行处理")):
    container = get_container()
    limit = container.settings.max_files_per_request
    # 先整批、后逐文件（04 §4.1）：超上限整批拒绝，一个都不处理
    if not files:
        raise AppError("VALIDATION_ERROR", "未收到文件")
    if len(files) > limit:
        raise AppError("TOO_MANY_FILES", f"每次最多上传 {limit} 个文件，请分批提交")

    items = []
    for upload in files:
        result = await documents.upload(container, upload)
        items.append(UploadItem(**result))

    # 关键词索引由 PG 的 GIN 倒排自动维护（V003），导入后无需重建，
    # 因此"导入成功"天然等于"立刻可检索"。

    # 逐文件结果按请求顺序返回；单文件失败不影响其它文件（同一响应里逐行反馈）
    return Envelope(data=UploadResult(items=items))


@router.get("", response_model=Envelope[DocumentList], summary="文档列表（全量，不分页）")
async def list_documents():
    result = documents.list_documents(get_container())
    return Envelope(data=DocumentList(**result))


@router.get("/{document_id}/chunks", response_model=Envelope[DocumentChunks], summary="文档片段预览（只读）")
def document_chunks(
    document_id: int,
    offset: int = Query(0, ge=0, description="从第几段开始"),
    limit: int = Query(20, ge=1, le=200, description="本次返回多少段（上限 200）"),
):
    result = documents.document_chunks(get_container(), document_id, offset, limit)
    return Envelope(data=DocumentChunks(**result))


@router.delete("/{document_id}", response_model=Envelope[None], summary="删除文档")
async def delete_document(document_id: int):
    await documents.delete_document(get_container(), document_id)
    return Envelope(data=None)
