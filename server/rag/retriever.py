"""相似度检索（M7）：query embedding → 全表 chunks 余弦 → 阈值 + Top-K。

bge 用法约束（批次 0 spike 实证）：query 侧加检索指令前缀（document 侧
不加）；向量 L2 归一化后点积即余弦。阈值的作用是拦域外 query——同域
错误文档的分数可能与命中交叠，靠 Top-K 兜底（见 spike 结论）。
"""
import math
import time
from dataclasses import dataclass

from config import config
from rag.embedder import embedder
from rag.ingest import deserialize_embedding
from session_store.sqlite_store import session_store


@dataclass
class SearchResult:
    hits: list[dict]  # [{chunk_id, document_id, filename, score, content}]
    latency_ms: float


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def search(query: str) -> SearchResult:
    """检索知识库；空库直接返回空命中（不做无谓的 embedding 调用）。"""
    t0 = time.perf_counter()
    chunks = session_store.get_all_chunks()
    if not chunks:
        return SearchResult(hits=[], latency_ms=_elapsed_ms(t0))

    filenames = {d.id: d.filename for d in session_store.list_documents()}
    query_vec = embedder.embed_texts([config.rag_query_prefix + query])[0]

    scored = []
    for chunk in chunks:
        # 幂等归一化保险：入库向量已归一化，点积即余弦
        vec = _l2_normalize(deserialize_embedding(chunk.embedding))
        score = sum(q * v for q, v in zip(query_vec, vec))
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    hits = [{"chunk_id": chunk.id,
             "document_id": chunk.document_id,
             "filename": filenames.get(chunk.document_id, "?"),
             "score": round(score, 4),
             "content": chunk.content}
            for score, chunk in scored
            if score >= config.rag_score_threshold][:config.rag_top_k]
    return SearchResult(hits=hits, latency_ms=_elapsed_ms(t0))


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
