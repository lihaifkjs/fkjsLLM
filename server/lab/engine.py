"""实验编排（M6 参数实验场）：同一 prompt 多组参数的单轮对比。

与 chat 平级（技术方案 1.1）：直调 model_manager / session_store / params，
不知道 HTTP。观测回填复刻 chat/engine._wrap 的 finally 汇合模式——
正常结束、被 stop、中途异常三路径都经 finish_run 落库，缺一路径
run 观测字段即残缺（M6 计划 §8）。
"""
import json
import threading
from dataclasses import dataclass
from typing import Iterator

from config import SamplingParams, config
from model_manager.manager import GenerateResult, ServiceError, model_manager
from params import presets
from session_store.base import Experiment, ExperimentRun, RunResult
from session_store.sqlite_store import session_store

MIN_PARAM_SETS = 2
MAX_PARAM_SETS = 3

RUN_INTERRUPTED = {"code": "RUN_INTERRUPTED",
                   "message": "生成被中断（服务重启或连接断开）"}


@dataclass
class RunMeta:
    experiment_id: int
    run_id: int


class LabEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_run_id: int | None = None  # 全局单路生成（与 chat 互斥）

    # ---------- 实验 ----------

    def create_experiment(self, prompt: str,
                          param_sets: list[dict]) -> Experiment:
        """校验后建实验（连 runs）；非法输入抛 ValueError（api 层转 400）。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        if not MIN_PARAM_SETS <= len(param_sets) <= MAX_PARAM_SETS:
            raise ValueError(
                f"参数组数必须在 {MIN_PARAM_SETS}–{MAX_PARAM_SETS} 之间")
        normalized = [presets.validate_run_params(ps) for ps in param_sets]
        # system prompt 快照取创建时全局值（决策 D：保证实验间可比性）
        system_prompt = presets.get_params().system_prompt
        return session_store.create_experiment(
            prompt, system_prompt,
            [json.dumps(ps, ensure_ascii=False) for ps in normalized])

    def get_experiment_detail(
            self, experiment_id: int) -> tuple[Experiment, list[ExperimentRun]]:
        """详情 = 实验 + 全部 runs。

        读路径纠正残留 running（计划 §8）：后端实际已断（重启/连接断开）
        的 run 永久显示「生成中」，此处纠正为 error 并落库。
        """
        exp = session_store.get_experiment(experiment_id)
        runs = session_store.get_runs(experiment_id)
        with self._lock:
            active = self._active_run_id
        fixed = []
        for r in runs:
            if r.status == "running" and r.id != active:
                session_store.finish_run(r.id, RunResult(
                    status="error", content=r.content,
                    error=json.dumps(RUN_INTERRUPTED, ensure_ascii=False)))
                r = session_store.get_run(experiment_id, r.id)
            fixed.append(r)
        return exp, fixed

    # ---------- 生成 ----------

    def run_generate(self, experiment_id: int, run_id: int
                     ) -> tuple[RunMeta, Iterator[str], GenerateResult]:
        """返回 (meta, token 迭代器, 结果回填对象)；迭代完（含被停止）落库。

        互抢锁（决策 E）：已有生成任务时抛 GENERATION_BUSY（api 层 409），
        run 保持原状态可重试；推理期的错误走 SSE error 帧。
        """
        exp = session_store.get_experiment(experiment_id)
        run = session_store.get_run(experiment_id, run_id)
        # generating 覆盖 chat 侧生成；_active_run_id 覆盖 lab 侧生成
        # （generate 是惰性生成器，锁在首个 token 才占用，需自查运行态）
        if model_manager.status().generating:
            raise ServiceError("GENERATION_BUSY", "已有生成任务进行中")
        with self._lock:
            if self._active_run_id is not None:
                raise ServiceError("GENERATION_BUSY", "已有生成任务进行中")
            session_store.start_run(run.id)
            self._active_run_id = run.id

        run_params = json.loads(run.params)
        # repetition_penalty 用全局当前值（决策 D），四项采样参数用本组快照
        params = SamplingParams(
            temperature=run_params["temperature"],
            top_p=run_params["top_p"],
            top_k=run_params["top_k"],
            max_tokens=run_params["max_tokens"],
            repetition_penalty=presets.get_params().repetition_penalty,
            system_prompt=exp.system_prompt,
        )
        messages = []
        if exp.system_prompt:
            messages.append({"role": "system", "content": exp.system_prompt})
        messages.append({"role": "user", "content": exp.prompt})

        meta = RunMeta(experiment_id=experiment_id, run_id=run.id)
        result = GenerateResult()
        tokens = model_manager.generate(messages, params, result)
        return meta, self._wrap(run, exp, tokens, params, result), result

    def _wrap(self, run: ExperimentRun, exp: Experiment,
              tokens: Iterator[str], params: SamplingParams,
              result: GenerateResult) -> Iterator[str]:
        parts: list[str] = []
        status = "done"
        error = None
        try:
            # 单轮无历史，不走滑窗（决策 F）：超长直接 PROMPT_TOO_LONG。
            # 检查先于 token 迭代——generate 是惰性生成器，此时锁尚未占用
            budget = config.n_ctx - params.max_tokens
            used = model_manager.count_tokens(exp.prompt)
            if exp.system_prompt:
                used += model_manager.count_tokens(exp.system_prompt)
            if used > budget:
                raise ServiceError(
                    "PROMPT_TOO_LONG",
                    f"prompt 超长（{used} tokens，上限 {budget}）")
            for text in tokens:
                parts.append(text)
                yield text
            if result.stopped:
                status = "stopped"
        except ServiceError as e:
            status, error = "error", {"code": e.code, "message": e.message}
            raise  # 交由 api 层转 error 帧
        except Exception as e:
            status, error = "error", {"code": "INTERNAL", "message": str(e)}
            raise
        finally:
            # 正常结束、被停止、异常三路径汇合：部分结果与观测数据落库
            session_store.finish_run(run.id, RunResult(
                status=status,
                content="".join(parts),
                error=json.dumps(error, ensure_ascii=False)
                if error is not None else None,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                prompt_eval_tokens=result.prompt_eval_tokens,
                prompt_text=result.prompt_text,
                prompt_messages=json.dumps(result.prompt_messages,
                                           ensure_ascii=False)
                if result.prompt_messages is not None else None,
                prompt_seg_tokens=json.dumps(result.seg_tokens)
                if result.seg_tokens is not None else None,
                ttft_ms=result.ttft_ms,
                prefill_ms=result.prefill_ms,
                decode_ms=result.decode_ms,
            ))
            with self._lock:
                self._active_run_id = None

    def stop(self, experiment_id: int, run_id: int):
        session_store.get_experiment(experiment_id)
        session_store.get_run(experiment_id, run_id)
        with self._lock:
            active = self._active_run_id
        if active != run_id:
            raise ServiceError(
                "NO_ACTIVE_GENERATION", f"run {run_id} 没有生成任务")
        model_manager.stop()


lab_engine = LabEngine()
