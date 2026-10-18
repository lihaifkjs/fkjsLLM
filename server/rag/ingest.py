"""文档上传入库编排（M7）：校验 → 切块 → embedding → 落库。

任何一步失败整体报错不落库：校验/切块/embedding 全部在写库前完成，
写库阶段（add_document → add_chunks）若异常则补偿删除文档行（级联清掉
已写入的部分 chunks）。

chunk_mode（M12.1）：window = M7 固定窗口（默认，行为不变）；article =
按条切块，供法律语料使用——要求文本每条以行首 `《法名》第X条` 开头。
"""
import struct

from config import config
from model_manager.manager import ServiceError
from rag.chunker import chunk_by_article, chunk_text
from rag.embedder import embedder
from session_store.base import Document
from session_store.sqlite_store import session_store

_ALLOWED_EXTENSIONS = (".txt", ".md")
_CHUNK_MODES = ("window", "article")


def serialize_embedding(vec: list[float]) -> bytes:
    """float32 小端字节流（与 chunks.embedding BLOB 约定一致）。"""
    return struct.pack(f"<{len(vec)}f", *vec)


def deserialize_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def ingest_document(filename: str, data: bytes,
                    chunk_mode: str = "window") -> Document:
    """上传编排入口；校验失败抛 ServiceError（api 层转 400）。"""
    if chunk_mode not in _CHUNK_MODES:
        raise ServiceError("INVALID_CHUNK_MODE",
                           f"未知切块模式：{chunk_mode}（可选 window/article）")
    if not filename.lower().endswith(_ALLOWED_EXTENSIONS):
        raise ServiceError("INVALID_FILE_TYPE",
                           f"仅支持 .txt/.md 文件：{filename}")
    if len(data) > config.rag_upload_max_bytes:
        raise ServiceError(
            "FILE_TOO_LARGE",
            f"文件大小 {len(data)} 字节，超过上限"
            f" {config.rag_upload_max_bytes} 字节")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ServiceError("INVALID_ENCODING",
                           "文件不是有效的 utf-8 文本（GBK 等编码请先转码）")

    chunks = chunk_by_article(text) if chunk_mode == "article" \
        else chunk_text(text)
    if not chunks:
        raise ServiceError(
            "EMPTY_DOCUMENT",
            "未切出任何片段：按条切块要求每条以行首《法名》第X条 开头")
    vectors = embedder.embed_texts(chunks)

    document = session_store.add_document(
        filename=filename, content=text, char_count=len(text),
        chunk_count=len(chunks))
    try:
        session_store.add_chunks(
            document.id,
            [(seq, chunk, serialize_embedding(vec))
             for seq, (chunk, vec) in enumerate(zip(chunks, vectors))])
    except Exception:
        session_store.delete_document(document.id)
        raise
    return document
