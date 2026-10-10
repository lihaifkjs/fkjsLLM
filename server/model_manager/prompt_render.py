"""chat template 渲染复现与分段 token 统计（M5 可观测性）。

llama-cpp 在 create_chat_completion 内部用 gguf 元数据的 Jinja2 模板渲染
messages，无公开 API 暴露渲染结果。这里按 llama.py 注册 handler 时的同款
构造重建 Jinja2ChatFormatter 复现渲染；tokenize 参数
（add_bos=False, special=True）也与 chat_formatter_to_chat_completion_handler
内部的调用对齐——保证 prompt 文本与 token 计数和真实推理输入一致。

注意：metadata / _model / Jinja2ChatFormatter 均为 llama-cpp-python 内部
实现，结论绑定 0.3.36，升级库时必须重验。
"""
from llama_cpp import Llama, llama_chat_format


def _build_formatter(llm: Llama) -> llama_chat_format.Jinja2ChatFormatter:
    """复刻 llama.py __init__ 中 chat_template.default handler 的构造。"""
    template = llm.metadata["tokenizer.chat_template"]
    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    eos_token = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos_token = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
    return llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token=eos_token,
        bos_token=bos_token,
        stop_token_ids=[eos_id],
    )


def count_text(llm: Llama, text: str) -> int:
    """与 llama_chat_format.py 中 prompt tokenize 同款参数（added_special=True）。"""
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))


def render_prompt(llm: Llama, messages: list[dict]) -> str:
    """渲染后的完整模型输入文本（含 <|im_start|> 等特殊 token 与生成引导符）。"""
    return _build_formatter(llm)(messages=messages).prompt


def count_prompt_tokens(llm: Llama, messages: list[dict]) -> int:
    """精确输入 token 数（账单口径）＝渲染后整段 tokenize。"""
    return count_text(llm, render_prompt(llm, messages))


def segment_tokens(llm: Llama, messages: list[dict]) -> dict:
    """分段 token 统计：system / history（滑窗保留的历史轮）/ current（当前输入）。

    前缀差分：逐前缀渲染，相邻前缀 token 数相减得每条消息的 token 开销。
    每个前缀渲染都带同样的生成引导符尾巴（<|im_start|>assistant\\n），差分
    时被消掉；它只计入第 0 条消息一段（system 存在时即 system 段），多出的
    几个 token 是生成引导符本身的开销。
    """
    formatter = _build_formatter(llm)
    normalized = _normalize_messages(messages)
    counts = [
        count_text(llm, formatter(messages=normalized[:k]).prompt)
        for k in range(1, len(normalized) + 1)
    ]
    per_message = [counts[0]] + [counts[k] - counts[k - 1]
                                 for k in range(1, len(counts))]

    has_system = messages[0]["role"] == "system"
    return {
        "system": per_message[0] if has_system else 0,
        "history": sum(per_message[1:-1]) if has_system
                   else sum(per_message[:-1]),
        "current": per_message[-1],
    }


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """按 Qwen3 模板逻辑预先剥掉历史 assistant 消息的 think 块。

    模板只对「最后一次提问之后」的 assistant 保留 think，更早的一律剥掉
    （split('</think>')[-1]）。直接逐前缀渲染时，同一 assistant 消息在
    短前缀里是「最后一条」think 被保留、在长前缀里变历史被剥掉，前缀
    差分会出现负值——先按最终渲染口径统一预处理，差分才单调。
    仅限无 tools 的对话路径（tools 分支模板另有逻辑，M9 再处理）。
    """
    last_query = max((i for i, m in enumerate(messages)
                      if m["role"] == "user"), default=-1)
    normalized = []
    for i, m in enumerate(messages):
        if (m["role"] == "assistant" and i <= last_query
                and "</think>" in m["content"]):
            m = {**m,
                 "content": m["content"].split("</think>")[-1].lstrip("\n")}
        normalized.append(m)
    return normalized
