"""会话管理路由：列表 / 新建 / 取消息 / 重命名 / 删除（技术方案 §3.2）
+ M5 消息观测数据端点 + M6.5 摘要压缩观测。

只做 HTTP ↔ session_store 调用转换；业务错误由全局 handler 转 409。
"""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model_manager.manager import ServiceError
from params.presets import get as get_params_dict
from session_store.base import Compression
from session_store.sqlite_store import session_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateRequest(BaseModel):
    title: str | None = None


class RenameRequest(BaseModel):
    title: str


def _session_dict(s) -> dict:
    return {"id": s.id, "title": s.title,
            "created_at": s.created_at, "updated_at": s.updated_at}


@router.get("")
def list_sessions():
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at}
            for s in session_store.list_sessions()]


@router.post("")
def create_session(req: CreateRequest):
    title = (req.title or "").strip() or "新会话"
    return _session_dict(session_store.create_session(title))


@router.get("/{session_id}/messages")
def get_messages(session_id: int):
    """消息列表（M6.5 起包一层）：summary_covered_upto 为当前生效摘要
    覆盖到的最大消息 id（id ≤ 它的消息细节已丢失）；滑窗策略或无摘要
    时为 None——切回滑窗标识即消失，压缩记录本身保留可回切。
    """
    covered_upto = None
    if get_params_dict()["context_strategy"] == "summary":
        latest = session_store.get_latest_compression(session_id)
        if latest is not None:
            covered_upto = latest.covered_upto
    return {
        "messages": [{"id": m.id, "role": m.role, "content": m.content,
                      "created_at": m.created_at}
                     for m in session_store.get_messages(session_id)],
        "summary_covered_upto": covered_upto,
    }


def _compression_payload(session_id: int, message_id: int) -> dict | None:
    """调用链（M6.5）：本消息触发的新压缩给完整记录；只复用既有摘要给
    active_id；会话无摘要为 None。"""
    triggered = session_store.get_compression_by_trigger(session_id,
                                                         message_id)
    if triggered is not None:
        return _compression_dict(triggered, triggered=True)
    latest = session_store.get_latest_compression(session_id)
    if latest is not None:
        return {"triggered": False, "active_id": latest.id}
    return None


def _compression_dict(c: Compression, triggered: bool) -> dict:
    return {
        "triggered": triggered,
        "id": c.id,
        "prev_summary": c.prev_summary,
        "source_messages": json.loads(c.source_messages),
        "source_prompt": c.source_prompt,
        "summary": c.summary,
        "covered_upto": c.covered_upto,
        "source_tokens": c.source_tokens,
        "summary_tokens": c.summary_tokens,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "timing": {"ttft_ms": c.ttft_ms, "prefill_ms": c.prefill_ms,
                   "decode_ms": c.decode_ms},
    }


def _loads(raw: str | None):
    return json.loads(raw) if raw is not None else None


@router.get("/{session_id}/messages/{message_id}/observability")
def get_observability(session_id: int, message_id: int):
    """M5：单条消息的观测数据（调试抽屉数据源，抽屉打开时按需加载）。

    消息不存在用 404（区别于业务冲突的 409 约定）；会话不存在沿用
    SESSION_NOT_FOUND（全局 handler 转 409）。
    """
    session_store.get_session(session_id)
    try:
        m = session_store.get_message(message_id)
    except ServiceError as e:
        raise HTTPException(404, {"code": e.code, "message": e.message})
    if m.session_id != session_id:
        raise HTTPException(404, {"code": "MESSAGE_NOT_FOUND",
                                  "message": f"会话 {session_id} 中不存在消息"
                                             f" {message_id}"})
    return {
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "prompt_eval_tokens": m.prompt_eval_tokens,
        "seg_tokens": _loads(m.prompt_seg_tokens),
        "timing": {"ttft_ms": m.ttft_ms, "prefill_ms": m.prefill_ms,
                   "decode_ms": m.decode_ms},
        "params_snapshot": _loads(m.params_snapshot),
        "prompt_text": m.prompt_text,
        "prompt_messages": _loads(m.prompt_messages),
        "compression": _compression_payload(session_id, message_id),
    }


@router.patch("/{session_id}")
def rename_session(session_id: int, req: RenameRequest):
    if not req.title.strip():
        raise HTTPException(400, {"code": "EMPTY_TITLE",
                                  "message": "标题不能为空"})
    return _session_dict(
        session_store.rename_session(session_id, req.title.strip()))


@router.delete("/{session_id}")
def delete_session(session_id: int):
    session_store.delete_session(session_id)
    return {"deleted": session_id}
