"""对话编排：上下文拼接、停止转发。

不知道 HTTP，也不直接碰模型实例——只经 model_manager 推理；
会话/消息持久化委托 session_store，自身只保留运行态（当前生成会话）。
"""
import threading
from dataclasses import dataclass
from typing import Iterator

from chat.context import fit_window
from config import SamplingParams, config
from model_manager.manager import GenerateResult, ServiceError, model_manager
from params.presets import get_params
from session_store.base import Message
from session_store.sqlite_store import session_store

DEFAULT_TITLE = "新会话"
TITLE_LEN = 20  # 一期标题取首条消息前 20 字（PRD 3.2 简化）


@dataclass
class ChatMeta:
    session_id: int
    user_message_id: int
    assistant_message_id: int
    title: str


class ChatEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_session_id: int | None = None  # 全局单路生成

    # ---------- 对话 ----------

    def chat(self, session_id: int | None,
             content: str) -> tuple[ChatMeta, Iterator[str], GenerateResult]:
        """登记用户消息并返回 (meta, token 迭代器, 结果回填对象)。

        迭代器消费完毕（含被停止）时，把已生成的部分 assistant 消息落库。
        """
        with self._lock:
            if session_id is None:
                title = content[:TITLE_LEN].strip() or DEFAULT_TITLE
                session = session_store.create_session(title)
            else:
                session = session_store.get_session(session_id)
                if session.title == DEFAULT_TITLE:
                    # 空会话（POST /api/sessions 建的）首条消息生成标题
                    title = content[:TITLE_LEN].strip() or DEFAULT_TITLE
                    session = session_store.rename_session(session.id, title)

            user_msg = session_store.add_message(session.id, "user", content)
            # 占位 assistant 消息：先插入拿到 id（meta 要用），生成完回填内容
            assistant_msg = session_store.add_message(
                session.id, "assistant", "")
            self._active_session_id = session.id

            prompt, params = self._build_prompt(session.id,
                                                assistant_msg.id)
            meta = ChatMeta(session_id=session.id,
                            user_message_id=user_msg.id,
                            assistant_message_id=assistant_msg.id,
                            title=session.title)

        result = GenerateResult()
        tokens = model_manager.generate(prompt, params, result)
        return meta, self._wrap(assistant_msg, tokens, result), result

    def _build_prompt(self, session_id: int,
                      exclude_message_id: int) -> tuple[list[dict], SamplingParams]:
        """拼上下文并滑窗截断；同时返回本次生效的推理参数。

        参数每次生成前从 params 模块读取——改参数立即生效，不重载模型。
        """
        history = [{"role": m.role, "content": m.content}
                   for m in session_store.get_messages(session_id)
                   if m.id != exclude_message_id]
        params = get_params()
        prompt = fit_window(params.system_prompt, history,
                            budget=config.n_ctx - params.max_tokens,
                            count_tokens=model_manager.count_tokens)
        return prompt, params

    def _wrap(self, assistant_msg: Message, tokens: Iterator[str],
              result: GenerateResult) -> Iterator[str]:
        parts: list[str] = []
        try:
            for text in tokens:
                parts.append(text)
                yield text
        finally:
            # 正常结束与被停止都走到这里：部分结果与 token 计数落库
            session_store.update_message(
                assistant_msg.id, "".join(parts),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens)
            with self._lock:
                self._active_session_id = None

    def stop(self, session_id: int):
        with self._lock:
            active = self._active_session_id
        if active != session_id:
            raise ServiceError(
                "NO_ACTIVE_GENERATION", f"会话 {session_id} 没有生成任务")
        model_manager.stop()


chat_engine = ChatEngine()
