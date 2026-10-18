"""全局配置：纯数据类，所有模块可读，不依赖任何模块。"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# M8：可加载模型注册表。原版 Qwen3-14B + 身份微调版 fkjs-14B，
# POST /api/model/load 可选 model_id 切换（切换 = unload 后再 load）。
MODELS: dict[str, dict] = {
    "qwen3-14b": {
        "path": str(PROJECT_ROOT / "Qwen3-14B-Q4_K_M.gguf"),
        "name": "Qwen3-14B 原版",
    },
    "fkjs-14b": {
        "path": str(PROJECT_ROOT / "fkjs-14b-q4_k_m.gguf"),
        "name": "fkjs-14B（身份微调）",
    },
    # 2026-10-17：abliterated 对照用，二者均验证可被 llama-cpp-python
    # 0.3.36 加载。注意 qwen38-9b 声明的上下文为 256K，实际仍受全局 n_ctx 约束。
    "qwen3-14b-abliterated": {
        "path": str(PROJECT_ROOT / "Qwen3-14B-abliterated-Q4_K_M.gguf"),
        "name": "Qwen3-14B Abliterated",
    },
    "qwen38-9b-heretic": {
        "path": str(PROJECT_ROOT / "Qwen3.8-9B-heretic-uncensored.Q6_K.gguf"),
        "name": "Qwen3.8-9B Heretic",
    },
    # M11 再训练产物：checkpoint-150（epoch 2.5）合并 -> f16 -> Q4_K_M
    "fkjs-14b-v2": {
        "path": str(PROJECT_ROOT / "fkjs-14b-v2-q4_k_m.gguf"),
        "name": "fkjs-14B v2（M11）",
    },
    # M12 领域微调：劳动法 SFT（71 条语料 / 3 epoch），分片合并 LoRA -> f16 -> Q4_K_M
    "law-14b": {
        "path": str(PROJECT_ROOT / "law-14b-q4_k_m.gguf"),
        "name": "law-14B（劳动法微调）",
    },
}
DEFAULT_MODEL_ID = "qwen3-14b"


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
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "0"))
    n_ctx: int = 8192
    host: str = "0.0.0.0"
    port: int = 8000
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "0") == "1"  # M10 简单登录门
    # M10：凭据配置文件写死（docs/M10账号系统.md），可用环境变量覆盖，
    # 格式 "user1:pass1,user2:pass2"
    auth_users: dict[str, str] = field(default_factory=lambda: dict(
        pair.split(":", 1)
        for pair in os.getenv("AUTH_USERS", "admin:fkjs-admin").split(",")
        if ":" in pair
    ))
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
    rag_article_max_chars: int = 450  # 按条切块：超此字数的条按款再拆（M12.1）
    rag_top_k: int = 3
    # 阈值只负责拦「域外 query」（域内对错由 top-k 兜底）。按条切块实测
    # （spikes/law_recall_result.json，2026-10-17）：域外最高 0.3788，
    # 命中 query 的 top-1 最低 0.565、未命中最高 0.686（两段重叠）。
    # 0.40 → 0.42：法条片段更短更同质，域外分数整体上抬，留出余量。
    rag_score_threshold: float = 0.42
    # 按条 chunk 比 M7 窗口 chunk 长：top-3 注入最大 671 字 ≈ 500-670 token，
    # 512 会砍掉第 3 条 → 放宽到 900（n_ctx 8192 下占比可接受）
    rag_max_inject_tokens: int = 900
    rag_upload_max_bytes: int = 262144  # 256KB
    rag_query_prefix: str = "为这个句子生成表示以用于检索相关文章："  # bge 检索指令前缀
    # M9 网页搜索工具常量（spike 定值，见 spikes/m9_web_search_spike.py）
    tool_search_top_n: int = 5        # 注入的结果条数上限
    tool_fetch_pages: int = 2         # 抓正文的条数（其余结果只有 snippet）
    tool_page_max_chars: int = 1500   # 单条正文截断（字）
    tool_inject_max_tokens: int = 768  # tool response 总 token 上限
    tool_search_timeout: int = 10     # curl 单次超时（秒）
    tool_search_retries: int = 2      # 撞墙/空结果重试次数
    tool_search_retry_delay: float = 2.0  # 重试间隔（秒）


config = Config()
