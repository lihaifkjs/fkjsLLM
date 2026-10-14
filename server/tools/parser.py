"""<tool_call> 解析与流式前缀判别（M9）。

llama-cpp-python 0.3.36 的 tool call 不走 message.tool_calls 字段，
以 <tool_call>{json}</tool_call> 文本混在 content 流中（spike 已确认）。
决策轮加 /no_think 后输出形态可判别：空 think 壳 + tool_call 块，
或直接正文。
"""
import json
import re
from dataclasses import dataclass

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# 流式判别用的前缀候选：/no_think 下 think 壳是空的，但空白数量由模型
# 生成、不保证逐字一致，故壳分段判别（开标签 → 空白壳体 → 闭标签）
_TOOL_MARK = "<tool_call>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# 壳体未闭合时的合法途中形态：纯空白 + 至多一个 </think> 的截断前缀
# （截断可能从 '<' 开始，逐层可选）
_SHELL_BODY_RE = re.compile(
    r"\s*(?:<(?:/(?:t(?:h(?:i(?:n(?:k>?)?)?)?)?)?)?)?\s*")


@dataclass
class ToolCallParse:
    """<tool_call> 块解析结果；valid=False 时其余字段为空。"""
    valid: bool
    name: str = ""
    arguments: str = ""  # 原始 arguments JSON 文本
    query: str = ""


def parse_tool_call(raw: str) -> ToolCallParse:
    """解析 <tool_call> 块内的 JSON（web_search 单工具，只认 query 参数）。"""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ToolCallParse(valid=False)
    args = obj.get("arguments") or {}
    if isinstance(args, str):  # 少数模型会输出字符串化的 arguments
        args_raw = args
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    else:
        args_raw = json.dumps(args, ensure_ascii=False)
    query = args.get("query") or ""
    return ToolCallParse(valid=True, name=obj.get("name") or "",
                         arguments=args_raw, query=query)


class ToolCallDetector:
    """流式前缀判别器：增量喂 token 文本，返回可下发给前端的文本。

    三态：
    - pending：缓冲内容仍是「空 think 壳 + <tool_call>」前缀链的候选，
      暂不吐出（tool_call 标记不外泄到气泡的前提是命中前不漏字）
    - text：已偏离前缀，按普通文本处理，缓冲回吐 + 之后原样透传
    - tool：确认进入 tool_call 块，静默吞掉直到流结束

    用法：逐 chunk feed()，流结束后调 finish() 取完整 tool_call 原文。
    """

    def __init__(self):
        self._buf = ""
        self.state = "pending"  # pending / text / tool

    def feed(self, chunk: str) -> str | None:
        """返回可吐给前端的文本；None 表示本次无可吐（缓冲或吞掉）。"""
        if self.state == "text":
            return chunk
        self._buf += chunk
        if self.state == "tool":
            return None
        return self._judge()

    def _judge(self) -> str | None:
        stripped = self._buf.lstrip()
        if not stripped:
            return None  # 只有空白，继续等
        # 缓冲仍是三个候选标签（<think> / </think> / <tool_call>）的截断
        # 前缀时不判别：llama-cpp 模板会把 <think> 预开进 prompt，流可能
        # 以 </think> 开头（空壳由模型补闭），< 与 </ 前缀必须继续等
        if any(tag.startswith(stripped)
               for tag in (_THINK_OPEN, _THINK_CLOSE, _TOOL_MARK)):
            return None
        if stripped.startswith(_THINK_OPEN):
            close = stripped.find(_THINK_CLOSE, len(_THINK_OPEN))
            if close == -1:
                # 壳未闭合：纯空白壳体（含闭标签截断前缀）继续等；
                # 出现真实思考内容说明 /no_think 失效，放弃判别转正文
                if _SHELL_BODY_RE.fullmatch(stripped[len(_THINK_OPEN):]):
                    return None
                return self._to_text()
            if stripped[len(_THINK_OPEN):close].strip():
                return self._to_text()  # 非空 think：正常文本路径
            stripped = stripped[close + len(_THINK_CLOSE):].lstrip()
        elif stripped.startswith(_THINK_CLOSE):
            # prompt 预开 think 壳：闭标签即空壳结束，后面才是正文/工具
            stripped = stripped[len(_THINK_CLOSE):].lstrip()
        if not stripped:
            return None  # 壳闭合后内容未到，继续等
        if _TOOL_MARK.startswith(stripped):
            return None  # 壳后 <tool_call> 的截断前缀，继续等
        if stripped.startswith(_TOOL_MARK):
            self.state = "tool"
            return None
        return self._to_text()  # 偏离前缀链：普通文本

    def _to_text(self) -> str:
        """偏离前缀链：缓冲原样回吐（含 think 壳），之后 chunk 原样透传。"""
        self.state = "text"
        out, self._buf = self._buf, ""
        return out

    def finish(self) -> tuple[str | None, str]:
        """流结束收尾。返回 (tool_call 块内原文 | None, 剩余待吐文本)。"""
        if self.state == "tool":
            m = TOOL_CALL_RE.search(self._buf)
            return (m.group(1) if m else self._buf), ""
        # pending 流中断（如被 stop）：缓冲的都是正文，回吐
        leftover, self._buf = self._buf, ""
        self.state = "text"
        return None, leftover
