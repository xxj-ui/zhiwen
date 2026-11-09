"""M1 文档管理 + M2 入库流水线（02 §2.3 / 03 §3.2）

处理顺序（不可调换）：
  1. 逐文件校验（扩展名 / 文件名 / 大小 / 空文件）—— 失败是"这一项失败"，不是请求失败
  2. 流式落盘到中转文件（超限即中断）
  3. 解析（空结果视为失败）→ 切分 → 向量化
  4. 原文件就位（**事务提交之前**）：local 走 rename，minio 走 put_object
  5. 一个事务内：upsert documents → 全量替换 chunks → 更新 chunk_count
  6. 提交后删旧原文件/旧对象（关键词索引由 PG 自动维护，无需失效）
失败路径：任一步失败 → 删除刚写入的原文件或对象、旧数据与旧原文件保持原样（BR-2）。
数据库完全不可用 → 抛 `DB_UNAVAILABLE`（请求级 503，不塞进 items[].error）。

并发：上传（含覆盖）与删除共用 `container.write_lock` 串行；解析/切分/向量化在锁外，
单请求内多个文件按提交顺序串行处理（04 §3.7）。
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import psycopg
from psycopg_pool import PoolClosed, PoolTimeout

from server.core import storage
from server.core.errors import ERROR_CODES, DependencyUnavailable, NotFoundError
from server.rag import embedder, loaders, splitter
from server.core.timeutil import to_local_iso

logger = logging.getLogger("zhiwen.document_service")

# 数据库侧故障统一映射成 503：连接失败 / 池超时 / 池已关闭
DB_ERRORS = (psycopg.OperationalError, PoolTimeout, PoolClosed)

ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".xlsx", ".csv"}
TABLE_SUFFIXES = {".xlsx", ".csv"}


def _item_error(filename: str, code: str, message: str | None = None) -> dict:
    return {
        "filename": filename,
        "status": "failed",
        "document_id": None,
        "chunks": 0,
        "overwritten": False,
        "error": {"code": code, "message": message or ERROR_CODES[code][1]},
    }


def _validate_filename(name: str) -> str | None:
    """返回错误码；None 表示通过（AC-9：拒绝路径分隔符与 `..`）。"""
    if not name or len(name) > 255:
        return "INVALID_FILENAME"
    if "/" in name or "\\" in name or ".." in name:
        return "INVALID_FILENAME"
    if name.strip(". ") == "":
        return "INVALID_FILENAME"
    return None


def _drain(src, dest: Path, max_bytes: int) -> int:
    return storage.save_stream(src, dest, max_bytes)


def _doc_item(row: dict) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "file_size": row["file_size"],
        "chunk_count": row["chunk_count"],
        "created_at": to_local_iso(row["created_at"]),
        "updated_at": to_local_iso(row["updated_at"]),
    }


async def upload(container: Any, upload_file: Any) -> dict:
    settings = container.settings
    filename = (upload_file.filename or "").strip()
    suffix = Path(filename).suffix.lower()

    bad = _validate_filename(filename)
    if bad:
        return _item_error(filename, bad)
    if suffix not in ALLOWED_SUFFIXES:
        return _item_error(filename, "UNSUPPORTED_FILE_TYPE", f"不支持的文件类型：{suffix or '（无扩展名）'}")

    # 中转文件：local 后端最后 rename 到 upload_dir；minio 后端上送后即删
    spool_dir = settings.spool_dir
    spool_dir.mkdir(parents=True, exist_ok=True)
    temp = storage.new_spool_path(spool_dir, suffix)

    try:
        size = await asyncio.to_thread(_drain, upload_file.file, temp, settings.max_file_size)
    except storage.FileTooLarge:
        return _item_error(filename, "FILE_TOO_LARGE")
    except OSError as exc:
        logger.warning("上传落盘失败 file=%s err=%s", filename, exc)
        temp.unlink(missing_ok=True)
        return _item_error(filename, "STORAGE_FAILED")
    if size == 0:
        temp.unlink(missing_ok=True)
        return _item_error(filename, "EMPTY_FILE")

    try:
        docs = await asyncio.to_thread(
            loaders.load,
            temp,
            ocr=settings.ocr_enabled,
            ocr_dpi=settings.ocr_dpi,
            ocr_min_chars=settings.ocr_min_chars,
            ocr_max_pages=settings.ocr_max_pages,
        )
    except Exception as exc:
        logger.info("解析失败 file=%s err=%s", filename, exc)
        temp.unlink(missing_ok=True)
        code = "TABLE_EMPTY" if suffix in TABLE_SUFFIXES else "PARSE_FAILED"
        return _item_error(filename, code)

    if not any((d.get("text") or "").strip() for d in docs):
        temp.unlink(missing_ok=True)
        code = "TABLE_EMPTY" if suffix in TABLE_SUFFIXES else "PARSE_FAILED"
        return _item_error(filename, code)

    chunks = await asyncio.to_thread(
        splitter.split_documents,
        docs,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.chunk_min_size,
    )
    texts = [c["text"] for c in chunks]
    try:
        vectors = await asyncio.to_thread(embedder.embed_documents, texts)
        container.model_state = "loaded"
    except Exception as exc:
        container.model_state = "failed"
        logger.error("向量化失败 file=%s err=%s", filename, exc)
        temp.unlink(missing_ok=True)
        return _item_error(filename, "EMBEDDING_FAILED")

    # 原文件就位（事务之前）：local 走原子 rename，minio 走 put_object
    try:
        stored_path = await asyncio.to_thread(container.store.publish, temp, suffix, filename)
    except (storage.StorageError, OSError) as exc:
        logger.error("原文件存储失败 file=%s err=%s", filename, exc)
        temp.unlink(missing_ok=True)
        return _item_error(filename, "STORAGE_FAILED")
    file_type = suffix.lstrip(".")

    existing: dict | None = None
    # 写阶段串行（04 §3.7 / 02 S-01）：解析、切分、向量化都在锁外，同名并发上传在这里排队，
    # 后到者命中已存在的行走覆盖，不会撞 uq_documents_filename_lower 抛 500。
    async with container.write_lock:
        try:
            with container.pool().connection() as conn:
                existing = container.documents.find_by_filename(filename, conn=conn)
                if existing is not None:
                    document_id = existing["id"]
                    container.documents.replace_meta(
                        document_id,
                        filename=filename,
                        stored_path=stored_path,
                        file_type=file_type,
                        file_size=size,
                        conn=conn,
                    )
                    container.chunks.delete_by_document(document_id, conn=conn)
                else:
                    document_id = container.documents.insert(
                        filename=filename,
                        stored_path=stored_path,
                        file_type=file_type,
                        file_size=size,
                        conn=conn,
                    )
                container.chunks.insert_for_document(document_id, filename, chunks, vectors, conn=conn)
                container.documents.update_chunk_count(document_id, len(chunks), conn=conn)
        except DB_ERRORS as exc:
            container.store.delete_ref(stored_path)
            logger.error("入库失败（数据库不可用） file=%s err=%s", filename, exc)
            raise DependencyUnavailable("DB_UNAVAILABLE") from exc
        except Exception:
            container.store.delete_ref(stored_path)
            logger.exception("入库失败 file=%s", filename)
            raise

        if existing is not None:
            container.store.delete_ref(existing["stored_path"])

    logger.info(
        "入库成功 file=%s doc_id=%s chunks=%s overwritten=%s size=%s",
        filename, document_id, len(chunks), existing is not None, size,
    )
    return {
        "filename": filename,
        "status": "success",
        "document_id": document_id,
        "chunks": len(chunks),
        "overwritten": existing is not None,
        "error": None,
    }


def document_chunks(container: Any, document_id: int, offset: int = 0, limit: int = 20) -> dict:
    """文档预览（FR-8）：按 chunk_index 顺序分页读出该文档的片段。

    只读、不加写锁。片段直接取自 rag_chunks.content，所以原文件
    丢失（例如 uploads/ 被清空）也能看——核对引用时这点很关键。
    """
    try:
        with container.pool().connection() as conn:
            doc = container.documents.get(document_id, conn=conn)
            if doc is None:
                raise NotFoundError("DOCUMENT_NOT_FOUND")
            total = container.chunks.count_for_document(document_id, conn=conn)
            rows = container.chunks.list_for_document(document_id, offset, limit, conn=conn)
    except DB_ERRORS as exc:
        raise DependencyUnavailable("DB_UNAVAILABLE") from exc

    return {
        "document": {
            "id": doc["id"],
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "file_size": doc["file_size"],
            "chunk_count": doc["chunk_count"],
            "updated_at": to_local_iso(doc["updated_at"]),
        },
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "chunk_id": row["id"],
                "chunk_index": row["chunk_index"],
                "page": row["page"],
                "content": row["content"],
            }
            for row in rows
        ],
    }


def list_documents(container: Any) -> dict:
    try:
        with container.pool().connection() as conn:
            rows = container.documents.list_all(conn=conn)
    except DB_ERRORS as exc:
        raise DependencyUnavailable("DB_UNAVAILABLE") from exc
    return {"items": [_doc_item(r) for r in rows]}


async def delete_document(container: Any, document_id: int) -> None:
    """删除文档；与上传/覆盖共用同一把写锁串行（04 §3.7）。"""
    doc: dict | None = None
    async with container.write_lock:
        try:
            with container.pool().connection() as conn:
                doc = container.documents.get(document_id, conn=conn)
                if doc is None:
                    raise NotFoundError("DOCUMENT_NOT_FOUND")
                container.chunks.delete_by_document(document_id, conn=conn)
                container.documents.delete(document_id, conn=conn)
        except DB_ERRORS as exc:
            raise DependencyUnavailable("DB_UNAVAILABLE") from exc

        container.store.delete_ref(doc["stored_path"])
    logger.info("删除成功 doc_id=%s file=%s", document_id, doc["filename"])
