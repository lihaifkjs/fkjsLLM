"""模型实例单例：加载/卸载状态机 + 流式推理 + 停止信号。

不知道会话与参数的存在（技术方案 1.1 约束）。
加载/卸载在工作线程执行，避免阻塞 FastAPI 事件循环。
"""
import gc
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from llama_cpp import Llama

from config import SamplingParams, config


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class ServiceError(Exception):
    """业务错误，api 层统一转成 409。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class GenerateResult:
    """generate() 结束时回填的元信息（usage 随流末帧才可得）。"""
    input_tokens: int | None = None
    output_tokens: int = 0
    stopped: bool = False


@dataclass
class ModelStatus:
    state: ModelState
    vram_used_mb: int | None = None  # CPU 版无法读取，恒为 None
    error: str | None = None
    generating: bool = False


class ModelManager:
    """状态机：unloaded → loading → loaded → unloading → unloaded，
    loaded 出错 → error → 可重新 load。非法迁移抛 ServiceError（409）。
    """

    def __init__(self):
        self._lock = threading.Lock()       # 保护状态机字段
        self._gen_lock = threading.Lock()   # 全局生成锁：同一时刻仅一路生成
        self._state = ModelState.UNLOADED
        self._error: str | None = None
        self._llm: Llama | None = None
        self._stop_event: threading.Event | None = None

    # ---------- 状态查询 ----------

    def status(self) -> ModelStatus:
        with self._lock:
            return ModelStatus(
                state=self._state,
                error=self._error,
                generating=self._stop_event is not None,
            )

    def _set_state(self, state: ModelState, error: str | None = None):
        with self._lock:
            self._state = state
            self._error = error

    # ---------- 加载 / 卸载 ----------

    def load(self) -> ModelState:
        with self._lock:
            if self._state not in (ModelState.UNLOADED, ModelState.ERROR):
                raise ServiceError(
                    "INVALID_STATE", f"当前状态 {self._state} 不能加载")
            self._state = ModelState.LOADING
            self._error = None
        threading.Thread(target=self._load_worker, daemon=True).start()
        return ModelState.LOADING

    def _load_worker(self):
        try:
            llm = Llama(
                model_path=config.model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=config.n_ctx,
                verbose=False,
            )
        except Exception as e:
            self._set_state(ModelState.ERROR, str(e))
            return
        with self._lock:
            self._llm = llm
            self._state = ModelState.LOADED

    def unload(self) -> ModelState:
        with self._lock:
            if self._state != ModelState.LOADED:
                raise ServiceError(
                    "INVALID_STATE", f"当前状态 {self._state} 不能卸载")
            if self._stop_event is not None:
                raise ServiceError(
                    "GENERATION_IN_PROGRESS", "生成中禁止卸载，请先停止生成")
            self._state = ModelState.UNLOADING
        threading.Thread(target=self._unload_worker, daemon=True).start()
        return ModelState.UNLOADING

    def _unload_worker(self):
        with self._lock:
            self._llm = None
        gc.collect()  # 触发 Llama 析构，释放内存/显存
        self._set_state(ModelState.UNLOADED)

    # ---------- 推理 ----------

    def generate(
        self,
        messages: list[dict],
        params: SamplingParams,
        result: GenerateResult,
    ) -> Iterator[str]:
        """返回 token 迭代器；每 token 检查停止信号，结束后回填 result。

        同步阻塞迭代器——由 api 层放入线程池消费，不碰事件循环。
        """
        if not self._gen_lock.acquire(blocking=False):
            raise ServiceError(
                "GENERATION_IN_PROGRESS", "已有生成任务进行中")
        with self._lock:
            if self._state != ModelState.LOADED or self._llm is None:
                self._gen_lock.release()
                raise ServiceError("MODEL_NOT_LOADED", "模型未加载")
            llm = self._llm
            self._stop_event = threading.Event()
            stop_event = self._stop_event
        try:
            yield from self._iterate(llm, messages, params, result, stop_event)
        finally:
            with self._lock:
                self._stop_event = None
            self._gen_lock.release()

    def _iterate(self, llm: Llama, messages: list[dict],
                 params: SamplingParams, result: GenerateResult,
                 stop_event: threading.Event) -> Iterator[str]:
        # 输入 token 数：流式接口不返回 usage，逐条 tokenize 求和近似
        # （不含 chat template 自身开销，够前端做超限提示用）
        result.input_tokens = sum(
            len(llm.tokenize(m["content"].encode("utf-8"), add_bos=False))
            for m in messages)
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repetition_penalty,
            max_tokens=params.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if stop_event.is_set():
                result.stopped = True
                break
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("delta", {}).get("content")
            if text:
                result.output_tokens += 1
                yield text

    def count_tokens(self, text: str) -> int:
        """用当前模型 tokenize 计数——滑窗截断（chat/context.py）调用。"""
        with self._lock:
            llm = self._llm
        if llm is None:
            raise ServiceError("MODEL_NOT_LOADED", "模型未加载")
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stop(self):
        with self._lock:
            event = self._stop_event
        if event is None:
            raise ServiceError("NO_ACTIVE_GENERATION", "当前没有生成任务")
        event.set()


model_manager = ModelManager()
