"""实验场路由（M6）：experiments CRUD + run 级 SSE 生成 / 停止。

只做 HTTP ↔ lab.engine 调用转换，不含业务逻辑（技术方案 1.1 约束）。
SSE 帧型与 chat 完全一致（meta/token/done/error，done 带 usage+timing）。
EXPERIMENT_NOT_FOUND / RUN_NOT_FOUND 用 404（对齐 M5 MESSAGE_NOT_FOUND），
区别于业务冲突的 409 约定；参数校验失败 400 INVALID_PARAMS。
"""
import json
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lab.engine import lab_engine
from model_manager.manager import ModelState, ServiceError, model_manager
from session_store.base import Experiment, ExperimentRun
from session_store.sqlite_store import session_store

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

_NOT_FOUND_CODES = ("EXPERIMENT_NOT_FOUND", "RUN_NOT_FOUND")
LIST_PROMPT_LEN = 50  # 列表接口 prompt 截取前 50 字（M6 计划 4.3）


class CreateExperimentRequest(BaseModel):
    prompt: str
    param_sets: list[dict]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _loads(raw: str | None):
    return json.loads(raw) if raw is not None else None


def _run_dict(r: ExperimentRun) -> dict:
    """run 全字段下发（决策 B：观测数据随详情接口一次取全）。"""
    return {
        "id": r.id, "experiment_id": r.experiment_id, "seq": r.seq,
        "params": _loads(r.params),
        "status": r.status, "content": r.content, "error": _loads(r.error),
        "created_at": r.created_at, "finished_at": r.finished_at,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        "prompt_eval_tokens": r.prompt_eval_tokens,
        "prompt_text": r.prompt_text,
        "prompt_messages": _loads(r.prompt_messages),
        "prompt_seg_tokens": _loads(r.prompt_seg_tokens),
        "ttft_ms": r.ttft_ms, "prefill_ms": r.prefill_ms,
        "decode_ms": r.decode_ms,
    }


def _detail_dict(exp: Experiment, runs: list[ExperimentRun]) -> dict:
    return {"id": exp.id, "prompt": exp.prompt,
            "system_prompt": exp.system_prompt, "created_at": exp.created_at,
            "runs": [_run_dict(r) for r in runs]}


def _stream(meta, tokens: Iterator[str], result) -> Iterator[str]:
    yield _sse("meta", {"experiment_id": meta.experiment_id,
                        "run_id": meta.run_id})
    try:
        for text in tokens:
            yield _sse("token", {"text": text})
        yield _sse("done", {
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            "timing": {
                "ttft_ms": result.ttft_ms,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
            },
        })
    except ServiceError as e:
        yield _sse("error", {"code": e.code, "message": e.message})
    except Exception as e:  # 推理崩溃也要以 error 帧收尾，不能让流裸断
        yield _sse("error", {"code": "INTERNAL", "message": str(e)})


@router.get("")
def list_experiments():
    return [{"id": e.id, "prompt": e.prompt[:LIST_PROMPT_LEN],
             "created_at": e.created_at}
            for e in session_store.list_experiments()]


@router.post("")
def create_experiment(req: CreateExperimentRequest):
    try:
        exp = lab_engine.create_experiment(req.prompt, req.param_sets)
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_PARAMS",
                                  "message": str(e)})
    return _detail_dict(exp, session_store.get_runs(exp.id))


@router.get("/{experiment_id}")
def get_experiment(experiment_id: int):
    try:
        exp, runs = lab_engine.get_experiment_detail(experiment_id)
    except ServiceError as e:
        raise HTTPException(404, {"code": e.code, "message": e.message})
    return _detail_dict(exp, runs)


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int):
    try:
        session_store.delete_experiment(experiment_id)
    except ServiceError as e:
        raise HTTPException(404, {"code": e.code, "message": e.message})
    return {"deleted": experiment_id}


@router.post("/{experiment_id}/runs/{run_id}/generate")
def generate(experiment_id: int, run_id: int):
    # 前置校验（对齐 chat）：模型未加载直接 409，避免 run 被置 running
    if model_manager.status().state != ModelState.LOADED:
        raise HTTPException(409, {"code": "MODEL_NOT_LOADED",
                                  "message": "模型未加载，请先加载模型"})
    try:
        meta, tokens, result = lab_engine.run_generate(experiment_id, run_id)
    except ServiceError as e:
        if e.code in _NOT_FOUND_CODES:
            raise HTTPException(404, {"code": e.code, "message": e.message})
        raise  # GENERATION_BUSY 等走全局 handler → 409
    return StreamingResponse(
        _stream(meta, tokens, result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{experiment_id}/runs/{run_id}/stop")
def stop(experiment_id: int, run_id: int):
    try:
        lab_engine.stop(experiment_id, run_id)
    except ServiceError as e:
        if e.code in _NOT_FOUND_CODES:
            raise HTTPException(404, {"code": e.code, "message": e.message})
        raise  # NO_ACTIVE_GENERATION 走全局 handler → 409
    return {"state": "stopping"}
