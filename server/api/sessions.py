"""会话管理路由：列表 / 新建 / 取消息 / 重命名 / 删除（技术方案 §3.2）
+ M5 消息观测数据端点 + M6.5 摘要压缩观测。

只做 HTTP ↔ session_store 调用转换；业务错误由全局 handler 转 409。
"""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chat.engine import chat_engine
from model_manager.manager import ServiceError
from session_store.base import Compression
from session_store.sqlite_store import session_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateRequest(BaseModel):
    title: str | None = None


class UpdateRequest(BaseModel):
    """部分更新：title / rag_enabled / tool_enabled 至少给一个。"""
    title: str | None = None
    rag_enabled: bool | None = None
    tool_enabled: bool | None = None  # M9 网页搜索工具开关


def _session_dict(s) -> dict:
    return {"id": s.id, "title": s.title,
            "created_at": s.created_at, "updated_at": s.updated_at,
            "rag_enabled": bool(s.rag_enabled),
            "tool_enabled": bool(s.tool_enabled)}


@router.get("")
def list_sessions():
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at,
             "rag_enabled": bool(s.rag_enabled),
             "tool_enabled": bool(s.tool_enabled)}
            for s in session_store.list_sessions()]


@router.post("")
def create_session(req: CreateRequest):
    title = (req.title or "").strip() or "新会话"
    return _session_dict(session_store.create_session(title))


@router.get("/{session_id}/messages")
def get_messages(session_id: int):
    """消息列表（M6.5 起包一层）：summary_covered_upto 为该会话最新压缩
    记录覆盖到的最大消息 id（id ≤ 它的消息细节已丢失）；存在压缩记录即
    返回，与当前策略无关（批次4：滑窗策略下也显示有损标识）。
    """
    latest = session_store.get_latest_compression(session_id)
    return {
        "messages": [{"id": m.id, "role": m.role, "content": m.content,
                      "created_at": m.created_at}
                     for m in session_store.get_messages(session_id)],
        "summary_covered_upto": latest.covered_upto if latest else None,
    }


@router.post("/{session_id}/compress")
def compress_session(session_id: int):
    """手动压缩全部历史（批次4）：成功返回本次 compressions 记录。

    会话不存在 404、空会话 400（同 observability 的 HTTPException 形状）；
    生成锁忙 / 模型未加载由 chat 层 ServiceError 走全局 handler 转 409。
    """
    try:
        record = chat_engine.compress_session(session_id)
    except ServiceError as e:
        if e.code == "SESSION_NOT_FOUND":
            raise HTTPException(404, {"code": e.code, "message": e.message})
        if e.code == "EMPTY_SESSION":
            raise HTTPException(400, {"code": e.code, "message": e.message})
        raise
    return _compression_dict(record, triggered=True)


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


def _retrieval_payload(session_id: int, message_id: int) -> dict | None:
    """调用链（M7）：会话启用检索且本轮有落库 → 完整记录（含无命中
    hits: []）；未启用 / 无记录（开关后开的老消息）→ None。"""
    if not session_store.get_session(session_id).rag_enabled:
        return None
    r = session_store.get_retrieval_by_trigger(session_id, message_id)
    if r is None:
        return None
    return {"query": r.query, "hits": json.loads(r.hits),
            "injected_tokens": r.injected_tokens,
            "latency_ms": r.latency_ms}


def _tool_call_payload(session_id: int, message_id: int) -> dict | None:
    """调用链（M9）：会话启用工具且本轮有落库 → 完整记录（含失败态）；
    未启用 / 无记录 → None（仿 retrieval 两态）。"""
    if not session_store.get_session(session_id).tool_enabled:
        return None
    t = session_store.get_tool_call_by_trigger(session_id, message_id)
    if t is None:
        return None
    try:
        arguments = json.loads(t.arguments)
    except json.JSONDecodeError:
        arguments = t.arguments  # parse_error 时原文下发
    return {"name": t.name, "arguments": arguments, "query": t.query,
            "status": t.status, "results": json.loads(t.results),
            "inject_tokens": t.inject_tokens, "latency_ms": t.latency_ms,
            "error": t.error}


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
        "retrieval": _retrieval_payload(session_id, message_id),
        "tool_call": _tool_call_payload(session_id, message_id),
    }


@router.patch("/{session_id}")
def update_session(session_id: int, req: UpdateRequest):
    if req.title is None and req.rag_enabled is None \
            and req.tool_enabled is None:
        raise HTTPException(400, {"code": "EMPTY_UPDATE",
                                  "message": "没有需要更新的字段"})
    if req.title is not None:
        if not req.title.strip():
            raise HTTPException(400, {"code": "EMPTY_TITLE",
                                      "message": "标题不能为空"})
        session_store.rename_session(session_id, req.title.strip())
    if req.rag_enabled is not None:
        session_store.set_rag_enabled(session_id, req.rag_enabled)
    if req.tool_enabled is not None:
        session_store.set_tool_enabled(session_id, req.tool_enabled)
    return _session_dict(session_store.get_session(session_id))


@router.delete("/{session_id}")
def delete_session(session_id: int):
    session_store.delete_session(session_id)
    return {"deleted": session_id}
