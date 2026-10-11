"""对话编排：上下文拼接（滑窗 / 摘要压缩）、停止转发。

不知道 HTTP，也不直接碰模型实例——只经 model_manager 推理；
会话/消息/压缩记录持久化委托 session_store，自身只保留运行态
（当前生成会话）。
"""
import json
import threading
from dataclasses import dataclass, fields
from typing import Iterator

from chat.context import (SUMMARY_PREFIX, fit_window, render_summary_request,
                          split_overflow, strip_think)
from config import SamplingParams, config
from model_manager.manager import GenerateResult, ServiceError, model_manager
from params.presets import snapshot as params_snapshot
from session_store.base import (Compression, CompressionInput, Message,
                                MessageObservability)
from session_store.sqlite_store import session_store

DEFAULT_TITLE = "新会话"
TITLE_LEN = 20  # 一期标题取首条消息前 20 字（PRD 3.2 简化）
# 手动压缩的 trigger_message_id 哨兵：无对应消息（id 从 1 自增），
# 也不会命中调用链的 get_compression_by_trigger 查询
MANUAL_TRIGGER_ID = 0


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

            prompt, params, snapshot = self._build_prompt(
                session.id, assistant_msg.id)
            meta = ChatMeta(session_id=session.id,
                            user_message_id=user_msg.id,
                            assistant_message_id=assistant_msg.id,
                            title=session.title)

        result = GenerateResult()
        tokens = model_manager.generate(prompt, params, result)
        return meta, self._wrap(assistant_msg, tokens, result, snapshot), result

    def _build_prompt(self, session_id: int,
                      exclude_message_id: int
                      ) -> tuple[list[dict], SamplingParams, str]:
        """拼上下文（按 context_strategy 分流滑窗/摘要）；返回生效参数与快照。

        参数每次生成前从 params 模块读取——改参数立即生效，不重载模型。
        """
        msgs = [m for m in session_store.get_messages(session_id)
                if m.id != exclude_message_id]
        snap = params_snapshot()
        params = SamplingParams(
            **{f.name: snap[f.name] for f in fields(SamplingParams)})
        budget = config.n_ctx - params.max_tokens
        if params.context_strategy == "summary":
            prompt = self._build_summary_context(
                session_id, msgs, params, budget, exclude_message_id)
            if prompt is not None:
                return prompt, params, json.dumps(snap, ensure_ascii=False)
        history = [{"role": m.role, "content": m.content} for m in msgs]
        prompt = fit_window(params.system_prompt, history,
                            budget=budget,
                            count_tokens=model_manager.count_tokens)
        return prompt, params, json.dumps(snap, ensure_ascii=False)

    # ---------- 摘要压缩（M6.5） ----------

    def _build_summary_context(
            self, session_id: int, msgs: list[Message],
            params: SamplingParams, budget: int,
            trigger_message_id: int) -> list[dict] | None:
        """摘要策略拼上下文；返回 None 表示无需处理或压缩失败（回退滑窗）。

        最终上下文 = system prompt + 摘要段 + 最近 N 轮原文（决策 E）；
        增量追加：只压缩 id > covered_upto 的新溢出消息（决策 B）。
        """
        count = model_manager.count_tokens
        total = (count(params.system_prompt) if params.system_prompt else 0) \
            + sum(count(m.content) for m in msgs)
        if total <= budget:
            return None  # 未超限：不动作
        cut = split_overflow([m.role for m in msgs],
                             config.summary_keep_rounds)
        if cut == 0:
            return None  # 历史不足 N 轮，没有可压缩的溢出
        overflow, kept = msgs[:cut], msgs[cut:]

        latest = session_store.get_latest_compression(session_id)
        new_msgs = [m for m in overflow
                    if latest is None or m.id > latest.covered_upto]
        if new_msgs:
            try:
                summary = self._run_compression(
                    session_id, latest, new_msgs, trigger_message_id)
            except ServiceError:
                summary = None  # 锁忙/模型未加载同样降级滑窗（决策 D 兜底）
            if summary is None:
                return None  # 压缩失败降级：对话不中断（决策 D 兜底）
        else:
            summary = latest.summary  # 无新增溢出：复用既有摘要

        # 兜底：摘要 + N 轮仍超预算（N 轮过大/摘要过长）→ 对 N 轮再滑窗
        summary_block = SUMMARY_PREFIX + summary
        kept_dicts = [{"role": m.role, "content": m.content} for m in kept]
        if count(params.system_prompt + summary_block) \
                + sum(count(m.content) for m in kept) > budget:
            sys_text = "\n\n".join(
                filter(None, [params.system_prompt, summary_block]))
            return fit_window(sys_text, kept_dicts, budget=budget,
                              count_tokens=count)

        prompt = []
        if params.system_prompt:
            prompt.append({"role": "system", "content": params.system_prompt})
        prompt.append({"role": "system", "content": summary_block})
        prompt.extend(kept_dicts)
        return prompt

    def _run_compression(self, session_id: int,
                         latest: Compression | None,
                         new_msgs: list[Message],
                         trigger_message_id: int) -> str | None:
        """跑一次压缩推理并落库 compressions；失败返回 None。

        ServiceError（锁忙 GENERATION_IN_PROGRESS / 模型未加载）原样上抛，
        由调用方决定降级（自动路径）还是报错（手动压缩）。
        """
        prev_summary = latest.summary if latest else ""
        source = [{"role": m.role, "content": m.content} for m in new_msgs]
        request = render_summary_request(prev_summary, source)
        comp_params = SamplingParams(
            temperature=config.summary_temperature,
            max_tokens=config.summary_max_tokens)
        result = GenerateResult()
        try:
            summary = strip_think(
                "".join(model_manager.generate(request, comp_params, result)))
        except ServiceError:
            raise
        except Exception:
            return None
        if not summary or result.stopped:
            return None  # 空摘要/被停止：不可用，降级滑窗
        count = model_manager.count_tokens
        session_store.add_compression(CompressionInput(
            session_id=session_id,
            trigger_message_id=trigger_message_id,
            prev_summary=prev_summary,
            source_messages=json.dumps(source, ensure_ascii=False),
            source_prompt=result.prompt_text or "",
            summary=summary,
            covered_upto=new_msgs[-1].id,  # 增量游标：实际纳入的最大消息 id
            source_tokens=sum(count(m.content) for m in new_msgs),
            summary_tokens=count(summary),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            ttft_ms=result.ttft_ms,
            prefill_ms=result.prefill_ms,
            decode_ms=result.decode_ms,
        ))
        return summary

    def compress_session(self, session_id: int) -> Compression:
        """手动压缩指定会话全部历史（与自动压缩共用 _run_compression 路径）。

        增量合并保持一致：prev_summary 取最新 compression，本次 source 为
        id > covered_upto 的全部消息，covered_upto 推进到最新一条消息 id；
        trigger_message_id 置 MANUAL_TRIGGER_ID 表示手动。
        空会话抛 EMPTY_SESSION（api 层转 400）；锁忙/模型未加载的
        ServiceError 原样上抛（409）；全部历史已覆盖时幂等返回现有记录。
        """
        msgs = session_store.get_messages(session_id)  # 不存在抛 SESSION_NOT_FOUND
        if not msgs:
            raise ServiceError("EMPTY_SESSION",
                               f"会话 {session_id} 没有消息，无可压缩历史")
        latest = session_store.get_latest_compression(session_id)
        new_msgs = [m for m in msgs
                    if latest is None or m.id > latest.covered_upto]
        if latest is not None and not new_msgs:
            return latest
        summary = self._run_compression(
            session_id, latest, new_msgs, MANUAL_TRIGGER_ID)
        if summary is None:
            raise ServiceError("COMPRESSION_FAILED", "压缩失败，请重试")
        return session_store.get_latest_compression(session_id)

    def _wrap(self, assistant_msg: Message, tokens: Iterator[str],
              result: GenerateResult, snapshot: str) -> Iterator[str]:
        parts: list[str] = []
        try:
            for text in tokens:
                parts.append(text)
                yield text
        finally:
            # 正常结束与被停止都走到这里：部分结果与观测数据落库
            session_store.update_message(
                assistant_msg.id, "".join(parts),
                MessageObservability(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    params_snapshot=snapshot,
                    prompt_text=result.prompt_text,
                    prompt_messages=json.dumps(result.prompt_messages,
                                               ensure_ascii=False)
                    if result.prompt_messages is not None else None,
                    prompt_seg_tokens=json.dumps(result.seg_tokens)
                    if result.seg_tokens is not None else None,
                    prompt_eval_tokens=result.prompt_eval_tokens,
                    ttft_ms=result.ttft_ms,
                    prefill_ms=result.prefill_ms,
                    decode_ms=result.decode_ms,
                ))
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
