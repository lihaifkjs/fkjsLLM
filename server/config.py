"""全局配置：纯数据类，所有模块可读，不依赖任何模块。"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SamplingParams:
    """推理参数默认值，对齐 Qwen3 官方推荐（见技术方案 3.3）。"""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_tokens: int = 2048
    system_prompt: str = ""


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
    sampling: SamplingParams = field(default_factory=SamplingParams)


config = Config()
