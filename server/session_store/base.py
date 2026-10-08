"""会话存储接口：数据结构与类型化签名约定（技术方案 1.1）。

按已定决策不用 abc.ABC 强制继承——实现类（sqlite_store.SqliteSessionStore）
结构对齐本文件的签名即可，换库只换实现。
接口方法清单：
  create_session(title) -> Session
  get_session(session_id) -> Session          # 不存在抛 SESSION_NOT_FOUND
  list_sessions() -> list[Session]            # 按 updated_at 倒序
  rename_session(session_id, title) -> Session
  delete_session(session_id)                  # 级联删消息
  add_message(session_id, role, content) -> Message
  update_message(message_id, content, input_tokens, output_tokens)
  get_messages(session_id) -> list[Message]   # 按 id 升序
  get_setting(key) -> str | None              # value 为 JSON 字符串
  set_setting(key, value)
"""
from dataclasses import dataclass


@dataclass
class Session:
    id: int
    title: str
    created_at: float
    updated_at: float


@dataclass
class Message:
    id: int
    session_id: int
    role: str  # user / assistant / system
    content: str
    created_at: float
    # 可观测性字段：token 计数随消息落库；params_snapshot 一期恒为 None（PRD 3.7）
    input_tokens: int | None = None
    output_tokens: int | None = None
    params_snapshot: str | None = None
