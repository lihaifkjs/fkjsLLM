"""切块（M7 固定窗口 / M12 按条）。

M7：固定窗口 + 重叠，窗口右界优先回退到段落边界。
M12：法律语料按「条」切——行首 `[《法律名》]第X条` 为分隔符，一条一块。
     法律名与条号作为文本前缀保留在 chunk 内（`《劳动合同法》第十九条：…`），
     注入侧直接把它当引用标签用，chunks 表 schema 无需变更。
     超过 max_chars 的条按「款」（源文本行）再拆，前缀带款序号。
"""
import re

from config import config

# 条首：可带法律名前缀；`第[中文数字]条`
_ARTICLE_HEAD_RE = re.compile(r"^(《[^》]{1,40}》)?第[零〇一二三四五六七八九十百]+条")
# 章/节标题与目录行：不参与切块（法律文本的导航信息，不是条文内容）
_CHAPTER_RE = re.compile(r"^第[零〇一二三四五六七八九十百]+[章节]")
_TOC_RE = re.compile(r"^目\s*录$")
# 已带引用标签的 chunk（注入侧据此判断是否自带出处）
_ARTICLE_CHUNK_RE = re.compile(r"^《[^》]{1,40}》第[零〇一二三四五六七八九十百]+条")


def is_article_chunk(content: str) -> bool:
    """chunk 是否自带 `《法》第X条：` 引用标签（按条切块的产物）。"""
    return bool(_ARTICLE_CHUNK_RE.match(content))


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


def chunk_by_article(text: str,
                     max_chars: int = config.rag_article_max_chars) -> list[str]:
    """按条切块：一条一块，块内保留「《法》第X条：」前缀。

    传入文本要求每条以行首 `[《法律名》]第X条` 开头（见
    server/rag/corpus/），条内的款项/分项沿用源文本换行。首个条首之前的
    前言、章节目录行丢弃——法律语料里它们不承载条文。
    """
    blocks: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _TOC_RE.match(line) or _CHAPTER_RE.match(line):
            continue
        head = _ARTICLE_HEAD_RE.match(line)
        if head:
            blocks.append((head.group(0), [line]))
        elif blocks:
            blocks[-1][1].append(line)
    chunks: list[str] = []
    for label, lines in blocks:
        chunks.extend(_split_long_article(label, lines, max_chars))
    return chunks


def _split_long_article(label: str, lines: list[str],
                        max_chars: int) -> list[str]:
    """单条超 max_chars 时按款再拆；拆分后前缀补「（第k/m款）」。"""
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buf = ""
    for line in lines:
        if buf and len(buf) + 1 + len(line) > max_chars:
            parts.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        parts.append(buf)
    total = len(parts)
    if total == 1:  # 单款仍超限：硬切，保证不会远超 embedding 窗口
        body = text[len(label):]
        pieces = [body[j:j + max_chars] for j in range(0, len(body), max_chars)]
        return [f"{label}（第{i}/{len(pieces)}段）{piece}"
                for i, piece in enumerate(pieces, 1)]
    out: list[str] = []
    for i, part in enumerate(parts, 1):
        body = part[len(label):] if part.startswith(label) else part
        out.append(f"{label}（第{i}/{total}款）{body}")
    return out
