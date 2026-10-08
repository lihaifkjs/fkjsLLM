"""会话管理路由：列表 / 新建 / 取消息 / 重命名 / 删除（技术方案 §3.2）。

只做 HTTP ↔ session_store 调用转换；业务错误由全局 handler 转 409。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    return [{"id": m.id, "role": m.role, "content": m.content,
             "created_at": m.created_at}
            for m in session_store.get_messages(session_id)]


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
