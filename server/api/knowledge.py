"""知识库文档管理路由（M7）：列表 / 上传 / 删除。

只做 HTTP ↔ rag 编排调用转换；ingest 抛出的 ServiceError 按 code 映射
400（客户端文件问题），其余走全局 handler 转 409；DOCUMENT_NOT_FOUND
用 404（同 observability 的消息 404 形状）。

上传可选 `chunk_mode` 表单字段（M12.1）：默认 window（M7 固定窗口），
article = 按条切块（法律语料）。前端不传即保持一期/二期行为不变。
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from model_manager.manager import ServiceError
from rag.ingest import ingest_document
from session_store.base import Document
from session_store.sqlite_store import session_store

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

_UPLOAD_ERRORS = ("INVALID_FILE_TYPE", "FILE_TOO_LARGE", "INVALID_ENCODING",
                  "INVALID_CHUNK_MODE", "EMPTY_DOCUMENT")


def _document_dict(d: Document) -> dict:
    return {"id": d.id, "filename": d.filename, "char_count": d.char_count,
            "chunk_count": d.chunk_count, "created_at": d.created_at}


@router.get("/documents")
def list_documents():
    return [_document_dict(d) for d in session_store.list_documents()]


@router.post("/documents")
async def upload_document(file: UploadFile = File(...),
                          chunk_mode: str = Form("window")):
    data = await file.read()
    try:
        document = ingest_document(file.filename or "", data, chunk_mode)
    except ServiceError as e:
        if e.code in _UPLOAD_ERRORS:
            raise HTTPException(400, {"code": e.code, "message": e.message})
        raise
    return _document_dict(document)


@router.delete("/documents/{document_id}")
def delete_document(document_id: int):
    try:
        session_store.delete_document(document_id)
    except ServiceError as e:
        if e.code == "DOCUMENT_NOT_FOUND":
            raise HTTPException(404, {"code": e.code, "message": e.message})
        raise
    return {"deleted": document_id}
