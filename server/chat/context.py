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

# RAG 检索结果注入主上下文时的段前缀（独立 system 段，M7 决策 E）
RAG_PREFIX = "以下是知识库中与当前问题相关的内容（供参考）：\n"

# M9 工具提示段：渲染期注入 system prompt 尾部（不写回 params，决策 L）；
# 措辞对齐 spikes/tool_calling_baseline.json 的验证基线
TOOL_SUFFIX_TMPL = "当前日期 {date}，你可以使用 web_search 工具获取实时信息。"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def fit_window(system_prompt: str, history: list[dict], budget: int,
               count_tokens: Callable[[str], int],
               extra_system: list[str] | None = None) -> list[dict]:
    """返回拼好的完整上下文（system → 固定附加段 → 裁剪后的历史）。

    extra_system：固定保留、不参与截断的附加 system 段（M7 RAG 注入段），
    位置在 system prompt 之后，token 计入 budget。
    count_tokens 由 model_manager 提供——tokenize 必须用模型实例本身。
    """
    extra_system = extra_system or []
    sys_tokens = (count_tokens(system_prompt) if system_prompt else 0) \
        + sum(count_tokens(s) for s in extra_system)
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
    prompt.extend({"role": "system", "content": s} for s in extra_system)
    prompt.extend(history[start:])
    return prompt


def render_rag_segment(hits: list[dict], max_tokens: int,
                       count_tokens: Callable[[str], int]) -> str | None:
    """检索命中渲染为注入段（M7）：前缀 + 各片段来源标注。

    hits 须已按分数降序；从高到低累加，超 max_tokens 丢弃后续片段；
    一个都放不下返回 None（调用方按无命中处理）。
    """
    parts: list[str] = []
    for hit in hits:
        candidate = RAG_PREFIX + "\n\n".join(
            parts + [f"[文档: {hit['filename']}]\n{hit['content']}"])
        if count_tokens(candidate) > max_tokens:
            break
        parts.append(f"[文档: {hit['filename']}]\n{hit['content']}")
    if not parts:
        return None
    return RAG_PREFIX + "\n\n".join(parts)


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


def render_tool_prompt_suffix() -> str:
    """M9：工具提示段（含当天日期），注入决策轮 system prompt 尾部。"""
    from datetime import date
    return TOOL_SUFFIX_TMPL.format(date=date.today().isoformat())


def render_tool_response(results: list[dict], max_tokens: int,
                         count_tokens: Callable[[str], int]) -> str:
    """搜索结果渲染为 tool 角色消息内容（M9）。

    正文（content）优先、snippet 兜底；按序累加，超 max_tokens 丢后续
    结果（上下文膨胀防护，PRD §8）；一个都放不下返回占位说明。
    """
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        body = r.get("content") or r.get("snippet") or ""
        block = f"[{i}] {r['title']}\n{r['url']}\n{body}".rstrip()
        candidate = "\n\n".join(parts + [block])
        if count_tokens(candidate) > max_tokens:
            break
        parts.append(block)
    return "\n\n".join(parts) if parts else "搜索无结果"
