"""embedding 单例（M7）：fastembed/onnxruntime 加载 bge-small-zh-v1.5，CPU。

批次 0 spike 选型结论（spikes/m7_embedding_spike.py）：fastembed 与
llama-cpp GGUF 命中率持平，耗时快 4-10 倍。独立 CPU 推理栈，不经过
model_manager（那是 14B chat 模型单例），也不碰生成锁。
"""
import math
import threading
from pathlib import Path

from fastembed import TextEmbedding

from config import config

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512
# 缓存固定在 server/data 下（与 app.db 同级，已被 .gitignore 忽略）：
# fastembed 0.9 默认把权重下到系统 TEMP，TEMP 被清理 / 沙箱受限时
# 会反复重下并失败（M12.1 实测坑点）。
CACHE_DIR = Path(config.db_path).parent / "fastembed"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class Embedder:
    """懒加载单例：首次 embed 时才下载/加载模型权重（约百 MB 内存常驻）。"""

    def __init__(self):
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def _load(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    self._model = TextEmbedding(model_name=MODEL_NAME,
                                                cache_dir=str(CACHE_DIR))
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """文本列表 → L2 归一化 float 向量列表（fastembed 输出已归一化，
        这里再归一化是幂等保险——spike 实测该行为不可依赖）。"""
        if not texts:
            return []
        model = self._load()
        return [_l2_normalize([float(x) for x in v])
                for v in model.embed(texts)]


embedder = Embedder()
