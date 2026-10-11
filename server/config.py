"""全局配置：纯数据类，所有模块可读，不依赖任何模块。"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SamplingParams:
    """推理参数默认值，对齐 Qwen3 官方推荐（见技术方案 3.3）。

    context_strategy：上下文超限处理策略（M6.5，决策 A 全局开关），
    sliding_window = 滑窗硬截断（M3），summary = 摘要压缩。
    """
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_tokens: int = 2048
    system_prompt: str = ""
    context_strategy: str = "sliding_window"


@dataclass(frozen=True)
class Config:
    # 当前装的是 CPU 版 llama-cpp-python，必须 0；换 CUDA 版后改 -1 全量 offload
    model_path: str = os.getenv(
        "MODEL_PATH", str(PROJECT_ROOT / "Qwen3-14B-Q4_K_M.gguf"))
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "0"))
    n_ctx: int = 8192
    host: str = "0.0.0.0"
    port: int = 8000
    auth_enabled: bool = False  # PRD 3.6 占位，一期不实现
    db_path: str = os.getenv(
        "DB_PATH", str(PROJECT_ROOT / "server" / "data" / "app.db"))
    # 调试：在 uvicorn 终端打印每次生成实际送入模型的 prompt 原文（拦截库内部
    # _create_completion，验证 think 剥离等模板行为用）；LOG_PROMPT=0 关闭
    log_prompt: bool = os.getenv("LOG_PROMPT", "1") == "1"
    sampling: SamplingParams = field(default_factory=SamplingParams)
    # M6.5 摘要压缩常量（spike 实测定值，见 spikes/summary_compression_spike.py）
    summary_keep_rounds: int = 4    # 摘要之外保留的最近对话轮数 N
    summary_max_tokens: int = 512   # 压缩调用的输出上限（含 think 块余量）
    summary_temperature: float = 0.2
    # M7 RAG 知识库常量（批次 0 spike 定值，见 spikes/m7_embedding_spike.py）
    rag_chunk_size: int = 500        # 切块窗口（字）
    rag_chunk_overlap: int = 50      # 切块重叠（字）
    rag_top_k: int = 3
    # 域外 query 最高 0.32 / 域内最低真实命中 0.4463 → 有效区间 [0.32, 0.44]
    rag_score_threshold: float = 0.40
    rag_max_inject_tokens: int = 512  # 注入段 token 上限
    rag_upload_max_bytes: int = 262144  # 256KB
    rag_query_prefix: str = "为这个句子生成表示以用于检索相关文章："  # bge 检索指令前缀


config = Config()
