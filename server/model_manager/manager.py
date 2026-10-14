"""模型实例单例：加载/卸载状态机 + 流式推理 + 停止信号。

不知道会话与参数的存在（技术方案 1.1 约束）。
加载/卸载在工作线程执行，避免阻塞 FastAPI 事件循环。
"""
import gc
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from llama_cpp import Llama, llama_cpp

from config import MODELS, DEFAULT_MODEL_ID, SamplingParams, config
from model_manager import prompt_render


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


def _wrap_prompt_logging(llm: Llama):
    """拦截库内部 create_completion/_create_completion，终端打印实际输入。

    chat handler 会先把渲染好的 prompt 自己 tokenize（llama_chat_format.py
    :658），所以这里收到的 prompt 通常是 token 列表，需 detokenize 还原；
    聊天与压缩调用同实例，都会被记录。
    """
    def _to_text(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        # special=True：保留 <|im_start|> 等特殊 token，还原完整模板结构
        return llm.detokenize(prompt, special=True).decode("utf-8", errors="replace")

    for name in ("create_completion", "_create_completion"):
        original = getattr(llm, name)

        def logged(prompt, *args, _orig=original, _name=name, **kwargs):
            text = _to_text(prompt)
            print(f"\n===== PROMPT[{_name}] ({len(text)} chars) =====\n"
                  f"{text}\n===== PROMPT end =====\n", flush=True)
            return _orig(prompt, *args, **kwargs)

        setattr(llm, name, logged)
    print("[prompt-log] wrapper installed", flush=True)


@dataclass
class GenerateResult:
    """generate() 结束时回填的元信息（M5：usage 精确值 + 性能计时）。

    input_tokens 为账单口径（渲染后整段 tokenize）；prompt_eval_tokens 为
    本轮实际 prefill 数（n_p_eval），KV 前缀复用下更小，差值即复用的算力。
    prefill_ms/decode_ms 取自 llama_perf_context（纯图计算耗时）；
    ttft_ms 是墙钟（含模板渲染与 tokenize 开销），三者口径不同属预期。
    """
    input_tokens: int | None = None
    output_tokens: int = 0
    stopped: bool = False
    prompt_text: str | None = None
    prompt_messages: list[dict] | None = None
    seg_tokens: dict | None = None
    prompt_eval_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class ModelStatus:
    state: ModelState
    vram_used_mb: int | None = None  # CPU 版无法读取，恒为 None
    error: str | None = None
    generating: bool = False
    # 当前选定的模型条目（未加载时也记忆，供下次 load 缺省使用）
    model_id: str = DEFAULT_MODEL_ID


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
        self._model_id = DEFAULT_MODEL_ID

    # ---------- 状态查询 ----------

    def status(self) -> ModelStatus:
        with self._lock:
            return ModelStatus(
                state=self._state,
                error=self._error,
                generating=self._stop_event is not None,
                model_id=self._model_id,
            )

    def _set_state(self, state: ModelState, error: str | None = None):
        with self._lock:
            self._state = state
            self._error = error

    # ---------- 加载 / 卸载 ----------

    def load(self, model_id: str | None = None) -> ModelState:
        with self._lock:
            if self._state not in (ModelState.UNLOADED, ModelState.ERROR):
                raise ServiceError(
                    "INVALID_STATE", f"当前状态 {self._state} 不能加载")
            if model_id is not None and model_id not in MODELS:
                raise ServiceError(
                    "UNKNOWN_MODEL", f"未知模型条目 {model_id}")
            self._model_id = model_id or self._model_id
            self._state = ModelState.LOADING
            self._error = None
            model_path = MODELS[self._model_id]["path"]
        threading.Thread(
            target=self._load_worker, args=(model_path,), daemon=True).start()
        return ModelState.LOADING

    def _load_worker(self, model_path: str):
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=config.n_ctx,
                verbose=False,
            )
        except Exception as e:
            self._set_state(ModelState.ERROR, str(e))
            return
        if config.log_prompt:
            _wrap_prompt_logging(llm)
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
        # 渲染完整 prompt 并精确计数（账单口径），同时按段统计 token 占比
        result.prompt_messages = messages
        result.prompt_text = prompt_render.render_prompt(llm, messages)
        result.input_tokens = prompt_render.count_text(llm, result.prompt_text)
        result.seg_tokens = prompt_render.segment_tokens(llm, messages)
        # verbose=False 时 llama-cpp 不自动重置计时，需手动——
        # 全局生成锁保证单路，perf 计数器无并发干扰
        llm._ctx.reset_timings()
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repetition_penalty,
            max_tokens=params.max_tokens,
            stream=True,
        )
        t0 = time.perf_counter()
        try:
            for chunk in stream:
                if stop_event.is_set():
                    result.stopped = True
                    break
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content")
                if text:
                    if result.ttft_ms is None:
                        result.ttft_ms = (time.perf_counter() - t0) * 1000
                    yield text
        finally:
            # 正常结束、被 stop、中途异常都尽力读 perf 计数器
            perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
            result.prefill_ms = perf.t_p_eval_ms
            result.decode_ms = perf.t_eval_ms
            result.prompt_eval_tokens = perf.n_p_eval
            # n_eval 是 decode 步数；首个生成 token 来自 prefill 的 logits
            # 不占 decode 步，故采样 token 总数 = n_eval + 1（发生过生成时），
            # 与 llama-cpp 非流式 usage.completion_tokens 口径一致
            result.output_tokens = (perf.n_eval + 1) if (
                perf.n_p_eval > 0 or perf.n_eval > 0) else 0

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
