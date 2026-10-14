"""360 搜索（so.com）抓取与正文提取（M9）。

选型实录（spikes/m9_web_search_spike.py，2026-10-14）：DDG 本机网络
不可达；Bing 按 TLS/IP 反爬投毒（httpx/curl_cffi 均失效）；百度/搜狗
反爬墙；360 低频可用、高频撞「访问异常页面」。故 HTTP 走 subprocess
系统 curl（SecureTransport 指纹实测通过率最高），撞墙重试，仍失败由
调用方落 search_error 降级（决策 B）。
2026-10-14 补：PC 端遭 IP 级封禁（真浏览器同撞验证码），移动端正
常——双端点互为兜底，解析器兼容两套页面结构（见 _parse_results）。
"""
import html as html_mod
import re
import subprocess
import time
from dataclasses import dataclass, field
from urllib.parse import quote

from config import config

WEB_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取实时信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    }
]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
             "Mobile/15E148 Safari/604.1")
# (搜索页, UA)，按序互为兜底：PC 撞 IP 级验证码时移动端仍可出结果
ENDPOINTS = [("https://www.so.com/s", UA), ("https://m.so.com/s", UA_MOBILE)]


@dataclass
class SearchOutcome:
    """一次搜索的结构化结果；error 非 None 即失败（results 为空）。"""
    results: list[dict] = field(default_factory=list)  # {title,url,snippet,content}
    latency_ms: float = 0.0
    error: str | None = None


def _curl(url: str, ua: str = UA) -> str:
    out = subprocess.run(
        ["curl", "-sL", "-m", str(config.tool_search_timeout),
         "-A", ua, url],
        capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="replace")


def _parse_results(page: str, max_results: int) -> list[dict]:
    """解析结果页 res-list 块，兼容 PC（li.res-list）与移动端
    （div.g-card.res-list）两套结构：
    - 真实 URL：PC 在 h3>a 的 data-mdurl，移动端在块级 data-pcurl
      （href 均为 so.com 跳转链，不可用）
    - 标题：h3 内文本（em/b 高亮标签随 clean 剥掉）
    - 摘要：class 含 res-desc（PC）或 summary（移动端）的首个 p/div
    """
    starts = [m.end() for m in
              re.finditer(r'<(?:li|div)[^>]*class="[^"]*res-list', page)]
    results = []
    clean = lambda s: html_mod.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(page)
        block = page[start:end]
        u = (re.search(r'data-pcurl="([^"]+)"', block) or
             re.search(r'data-mdurl="([^"]+)"', block))
        t = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.DOTALL)
        if not u or not t:
            continue  # 推荐卡等非结果块（无真实 URL）在此滤掉
        p = re.search(r'<(?:p|div)[^>]*class="[^"]*(?:res-desc|summary)[^"]*"'
                      r"[^>]*>(.*?)</(?:p|div)>", block, re.DOTALL)
        results.append({"title": clean(t.group(1)), "url": u.group(1),
                        "snippet": clean(p.group(1)) if p else "",
                        "content": None})
        if len(results) >= max_results:
            break
    return results


def _fetch_content(url: str, max_chars: int) -> str | None:
    """抓正文并提取（trafilatura），失败返回 None——snippet 兜底。"""
    import trafilatura  # 延迟导入：库较重，只在真正搜索时加载
    try:
        text = trafilatura.extract(_curl(url))
    except Exception:
        return None
    if not text:
        return None
    return text[:max_chars]


def web_search(query: str) -> SearchOutcome:
    """搜索 + Top-N 正文提取；任何异常收敛为 error 字段，不抛给编排层。"""
    t0 = time.perf_counter()
    try:
        results = []
        # 撞「访问异常页面」/ 空结果重试（spike 实测重试可挽回一部分）；
        # 每轮依次尝试 PC / 移动双端点（PC 遭 IP 级封禁时移动端兜底）
        for attempt in range(1 + config.tool_search_retries):
            for url, ua in ENDPOINTS:
                page = _curl(f"{url}?q={quote(query)}", ua)
                results = _parse_results(page, config.tool_search_top_n)
                if results:
                    break
            if results:
                break
            if attempt < config.tool_search_retries:
                time.sleep(config.tool_search_retry_delay)
        if not results:
            return SearchOutcome(
                latency_ms=(time.perf_counter() - t0) * 1000,
                error="搜索无结果或被反爬拦截")
        for r in results[:config.tool_fetch_pages]:
            r["content"] = _fetch_content(r["url"], config.tool_page_max_chars)
        return SearchOutcome(
            results=results, latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return SearchOutcome(latency_ms=(time.perf_counter() - t0) * 1000,
                             error=f"{type(e).__name__}: {e}")
