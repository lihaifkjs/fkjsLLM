"""上下文滑窗截断（技术方案 §5.3）。

总 token 超预算（n_ctx - max_tokens）时，从最早的 user/assistant
对话对开始丢弃；system prompt 与最新一条（当前输入）始终保留。
"""
from typing import Callable


def fit_window(system_prompt: str, history: list[dict], budget: int,
               count_tokens: Callable[[str], int]) -> list[dict]:
    """返回拼好的完整上下文（system → 裁剪后的历史）。

    count_tokens 由 model_manager 提供——tokenize 必须用模型实例本身。
    """
    sys_tokens = count_tokens(system_prompt) if system_prompt else 0
    counts = [count_tokens(m["content"]) for m in history]

    start = 0
    # 至少保留最后一条（当前输入）；成对丢最早的 user/assistant，
    # 只剩两条仍超限时退化为单条丢，保证当前输入不丢
    while len(history) - start > 1 \
            and sys_tokens + sum(counts[start:]) > budget:
        start += 2 if len(history) - start > 2 else 1

    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.extend(history[start:])
    return prompt
