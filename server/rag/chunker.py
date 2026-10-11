"""固定窗口 + 重叠切块纯函数（M7）：段落边界优先，utf-8 文本入。"""
from config import config


def chunk_text(text: str, size: int = config.rag_chunk_size,
               overlap: int = config.rag_chunk_overlap) -> list[str]:
    """按 size 字窗口切块，相邻块重叠 overlap 字。

    窗口右界落在文档中段时，优先回退到窗口内最后一个段落边界（\\n\\n）
    处切，避免把段落拦腰截断。
    """
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            boundary = text.rfind("\n\n", start + overlap, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1) if end < n else n
    return chunks
