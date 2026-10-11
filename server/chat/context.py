"""上下文管理（技术方案 §5.3）：滑窗硬截断（M3）+ 摘要压缩（M6.5）。

滑窗：总 token 超预算（n_ctx - max_tokens）时，从最早的 user/assistant
对话对开始丢弃；system prompt 与最新一条（当前输入）始终保留。

摘要压缩：本模块只提供纯函数（溢出切分 / 压缩模板渲染 / think 剥离），
压缩推理的编排（查缓存、调模型、落库、兜底降级）在 chat/engine.py。
"""
import re
from typing import Callable

# M6.5 压缩模板（措辞经 spikes/summary_compression_spike.py 实测调优）
SUMMARY_SYSTEM = (
    "你是对话压缩器。将【既有摘要】与【新增对话】合并为一份连贯摘要，"
    "保留关键事实、用户偏好、结论与待办，丢弃寒暄与细节复述。"
)
SUMMARY_USER_TMPL = (
    "【既有摘要】\n{prev_summary}\n\n"
    "【新增对话】\n{source}\n\n"
    "【要求】仅输出合并后的摘要，不超过 {max_chars} 字。"
)
# 摘要注入主上下文时的段前缀（独立 system 段，决策 E）
SUMMARY_PREFIX = "以下是此前对话的摘要（细节可能有损）：\n"
SUMMARY_MAX_CHARS = 400

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


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


def split_overflow(roles: list[str], keep_rounds: int) -> int:
    """返回切分点下标：[0, cut) 为溢出待压缩段，[cut:] 为最近 N 轮原文。

    切点对齐到 user 消息，保证保留段以 user 开头、溢出段以 assistant
    收尾；没有可压缩的溢出（历史不超过 N 轮 + 当前输入）时返回 0。
    """
    keep = keep_rounds * 2
    if len(roles) <= keep + 1:
        return 0
    cut = len(roles) - keep
    while cut < len(roles) - 1 and roles[cut] != "user":
        cut += 1
    return cut


def render_source(messages: list[dict]) -> str:
    """待压缩消息渲染为纯文本（role: content 逐行），供压缩模板与落库。"""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def render_summary_request(prev_summary: str,
                           source_messages: list[dict]) -> list[dict]:
    """渲染压缩调用的 messages（system 指令 + user 载荷）。"""
    user = SUMMARY_USER_TMPL.format(
        prev_summary=prev_summary or "（无）",
        source=render_source(source_messages),
        max_chars=SUMMARY_MAX_CHARS,
    )
    # /no_think：Qwen3 模板软开关，压缩不需要思考块（spike 实测可省约一半耗时）；
    # 产出的空 <think></think> 壳由 strip_think 剥掉
    user += " /no_think"
    return [{"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    """剥离 Qwen3 think 块；压缩输出只要正文。"""
    return _THINK_RE.sub("", text).strip()
