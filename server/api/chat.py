"""对话路由：POST /api/chat（SSE 流）+ POST /api/chat/stop。

只做 HTTP ↔ 内部调用转换，不含业务逻辑（技术方案 1.1 约束）。
EventSource 不支持 POST，前端用 fetch + ReadableStream 解析本 SSE 流。
"""
import json
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.engine import chat_engine
from model_manager.manager import ModelState, ServiceError, model_manager

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: int | None = None
    content: str


class StopRequest(BaseModel):
    session_id: int


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream(meta, tokens: Iterator[str], result) -> Iterator[str]:
    yield _sse("meta", {
        "session_id": meta.session_id,
        "user_message_id": meta.user_message_id,
        "assistant_message_id": meta.assistant_message_id,
        "title": meta.title,
    })
    try:
        for text in tokens:
            yield _sse("token", {"text": text})
        yield _sse("done", {"usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }})
    except ServiceError as e:
        yield _sse("error", {"code": e.code, "message": e.message})
    except Exception as e:  # 推理崩溃也要以 error 帧收尾，不能让流裸断
        yield _sse("error", {"code": "INTERNAL", "message": str(e)})


@router.post("")
def chat(req: ChatRequest):
    if not req.content.strip():
        raise HTTPException(400, {"code": "EMPTY_CONTENT",
                                  "message": "消息内容不能为空"})
    # 前置校验，避免未加载时污染会话历史（推理期的错误走 error 帧）
    if model_manager.status().state != ModelState.LOADED:
        raise HTTPException(409, {"code": "MODEL_NOT_LOADED",
                                  "message": "模型未加载，请先加载模型"})
    meta, tokens, result = chat_engine.chat(req.session_id, req.content)
    return StreamingResponse(
        _stream(meta, tokens, result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/stop")
def stop(req: StopRequest):
    chat_engine.stop(req.session_id)  # 无生成任务时抛 ServiceError → 409
    return {"state": "stopping"}
