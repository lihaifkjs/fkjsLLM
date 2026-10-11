"""M7 RAG 知识库：切块 / embedding / 入库编排 / 检索（批次 2 加 retriever）。

分层约束（技术方案 §1.1）：api → rag（文档管理）；chat → rag（检索）；
rag → session_store。embedder 为独立 CPU 推理栈，不经过 model_manager。
"""
