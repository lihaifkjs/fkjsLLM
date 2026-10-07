"""对话编排：上下文拼接、内存态会话（M1）、停止转发。

不知道 HTTP，也不直接碰模型实例——只经 model_manager 推理。
SQLite 持久化在 M3 以 session_store 模块替换内存实现。
"""
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from config import config
from model_manager.manager import GenerateResult, ServiceError, model_manager

DEFAULT_TITLE = "新会话"
TITLE_LEN = 20  # 一期标题取首条消息前 20 字（PRD 3.2 简化）


@dataclass
class ChatMeta:
    session_id: int
    user_message_id: int
    assistant_message_id: int
    title: str


class _Session:
    def __init__(self, session_id: int):
        self.id = session_id
        self.title = DEFAULT_TITLE
        self.messages: list[dict] = []  # [{"id","role","content","created_at"}]
        self.updated_at = time.time()


class ChatEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[int, _Session] = {}
        self._next_session_id = 1
        self._next_message_id = 1
        self._active_session_id: int | None = None  # 全局单路生成

    # ---------- 会话（M1 内存实现，接口形状对齐未来 session_store） ----------

    def list_sessions(self) -> list[dict]:
        with self._lock:
            sessions = sorted(self._sessions.values(),
                              key=lambda s: s.updated_at, reverse=True)
            return [{"id": s.id, "title": s.title, "updated_at": s.updated_at}
                    for s in sessions]

    def get_messages(self, session_id: int) -> list[dict]:
        return list(self._get_session(session_id).messages)

    def _get_session(self, session_id: int) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ServiceError("SESSION_NOT_FOUND", f"会话 {session_id} 不存在")
        return session

    # ---------- 对话 ----------

    def chat(self, session_id: int | None,
             content: str) -> tuple[ChatMeta, Iterator[str], GenerateResult]:
        """登记用户消息并返回 (meta, token 迭代器, 结果回填对象)。

        迭代器消费完毕（含被停止）时，把已生成的部分 assistant 消息落库。
        """
        with self._lock:
            if session_id is None:
                session = _Session(self._next_session_id)
                self._sessions[session.id] = session
                self._next_session_id += 1
            else:
                session = self._sessions.get(session_id)
                if session is None:
                    raise ServiceError(
                        "SESSION_NOT_FOUND", f"会话 {session_id} 不存在")

            user_msg = self._append(session, "user", content)
            if session.title == DEFAULT_TITLE:
                session.title = content[:TITLE_LEN] or DEFAULT_TITLE
            assistant_msg = self._append(session, "assistant", "")
            self._active_session_id = session.id

            # 拼上下文：system prompt（若配置）→ 历史 → 当前输入
            prompt = []
            if config.sampling.system_prompt:
                prompt.append({"role": "system",
                               "content": config.sampling.system_prompt})
            prompt.extend({"role": m["role"], "content": m["content"]}
                          for m in session.messages[:-1])  # 去掉占位 assistant

            meta = ChatMeta(session_id=session.id,
                            user_message_id=user_msg["id"],
                            assistant_message_id=assistant_msg["id"],
                            title=session.title)

        result = GenerateResult()
        tokens = model_manager.generate(
            prompt, config.sampling, result)
        return meta, self._wrap(session, assistant_msg, tokens, result), result

    def _wrap(self, session: _Session, assistant_msg: dict,
              tokens: Iterator[str], result: GenerateResult) -> Iterator[str]:
        parts: list[str] = []
        try:
            for text in tokens:
                parts.append(text)
                yield text
        finally:
            # 正常结束与被停止都走到这里：部分结果保留入库
            assistant_msg["content"] = "".join(parts)
            session.updated_at = time.time()
            with self._lock:
                self._active_session_id = None

    def stop(self, session_id: int):
        with self._lock:
            active = self._active_session_id
        if active != session_id:
            raise ServiceError(
                "NO_ACTIVE_GENERATION", f"会话 {session_id} 没有生成任务")
        model_manager.stop()

    def _append(self, session: _Session, role: str, content: str) -> dict:
        msg = {"id": self._next_message_id, "role": role,
               "content": content, "created_at": time.time()}
        self._next_message_id += 1
        session.messages.append(msg)
        return msg


chat_engine = ChatEngine()
