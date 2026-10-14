"""Spike: M9 网页搜索选型与 tool 二轮渲染验证。

用法:  server/.venv/bin/python server/spikes/m9_web_search_spike.py
结果:  server/spikes/m9_web_search_result.json

三部分：
1. 360 搜索（so.com）抓取定值：curl + 正则解析 res-list 块，测延迟/条数/结构
   （选型实录 2026-10-14：DDG 本机网络不可达；Bing 按 TLS/IP 反爬投毒——
   httpx、curl_cffi、甚至连续请求后的系统 curl 都拿到标题正确但结果无关的
   页面；百度首请求成功随后 1.4KB 验证页；搜狗 5KB 反爬墙；仅 360 三连稳定）
2. 正文提取对比：snippet vs curl+trafilatura 抓 Top-2 正文，定裁剪常量
3. 模型验证：tool 角色消息渲染结构 / prompt_render 在 tools= 下口径偏差 /
   真实搜索结果的二轮回答质量
"""
import html as html_mod
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))  # 复用 model_manager.prompt_render

MODEL_PATH = str(SERVER_DIR.parent / "Qwen3-14B-Q4_K_M.gguf")
RESULT_PATH = Path(__file__).with_name("m9_web_search_result.json")

SEARCH_URL = "https://www.so.com/s"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT = 10.0

TOOLS = [
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

QUERIES = ["最新一代 iPhone 的价格", "2026年诺贝尔文学奖得主", "llama.cpp 最新版本号"]


def _curl(url: str) -> str:
    """subprocess 调系统 curl 抓取（统一 HTTP 路径，指纹经实测可用）。"""
    out = subprocess.run(
        ["curl", "-sL", "-m", str(int(TIMEOUT)), "-A", UA, url],
        capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="replace")


def web_search(query: str, max_results: int = 5) -> tuple[list[dict], float]:
    """抓 360 搜索 HTML 结果页，返回 ([{title, url, snippet}], 耗时秒)。

    res-list 块按起始位置切片（内部有嵌套标签，不能用非贪婪闭合）；
    href 是 so.com 跳转链，真实 URL 取 a 标签的 data-mdurl 属性。
    """
    t0 = time.perf_counter()
    page = _curl(f"{SEARCH_URL}?q={quote(query)}")
    latency = time.perf_counter() - t0
    starts = [m.end() for m in
              re.finditer(r'<li[^>]*class="[^"]*res-list', page)]
    results = []
    clean = lambda s: html_mod.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(page)
        block = page[start:end]
        m = re.search(r'<h3[^>]*>\s*<a[^>]*data-mdurl="([^"]+)"[^>]*>(.*?)</a>',
                      block, re.DOTALL)
        if not m:
            continue
        p = re.search(r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>',
                      block, re.DOTALL)
        results.append({"title": clean(m.group(2)), "url": m.group(1),
                        "snippet": clean(p.group(1)) if p else ""})
        if len(results) >= max_results:
            break
    return results, latency


def part1_search() -> dict:
    out = []
    for q in QUERIES:
        try:
            results, latency = web_search(q)
            out.append({"query": q, "ok": bool(results), "count": len(results),
                        "latency_s": round(latency, 2),
                        "sample": results[:2],
                        "snippet_chars": [len(r["snippet"]) for r in results]})
            print(f"[p1] {'OK ' if results else 'EMPTY'} {q}: "
                  f"{len(results)} 条, {latency:.2f}s")
        except Exception as e:
            out.append({"query": q, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
            print(f"[p1] ERR {q}: {e}")
    return {"cases": out}


def part2_extract(first_results: list[dict]) -> dict:
    """对第一个查询的 Top-2 URL 跑 trafilatura，对比 snippet 与正文体量。"""
    import trafilatura
    out = []
    for r in first_results[:2]:
        t0 = time.perf_counter()
        try:
            downloaded = _curl(r["url"])  # 与搜索同一条 curl 路径，指纹一致
            text = trafilatura.extract(downloaded) if downloaded else None
            latency = time.perf_counter() - t0
            out.append({"url": r["url"][:100], "ok": bool(text),
                        "content_chars": len(text) if text else 0,
                        "snippet_chars": len(r["snippet"]),
                        "latency_s": round(latency, 2),
                        "content_head": (text or "")[:200]})
            print(f"[p2] {'OK ' if text else 'EMPTY'} {r['url'][:60]}: "
                  f"{len(text) if text else 0} 字, {latency:.2f}s")
        except Exception as e:
            out.append({"url": r["url"][:100], "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "latency_s": round(time.perf_counter() - t0, 2)})
            print(f"[p2] ERR {r['url'][:60]}: {e}")
    return {"cases": out}


def part3_model(search_query: str, search_results: list[dict]) -> dict:
    from llama_cpp import Llama
    from model_manager import prompt_render

    print(f"[p3] 加载模型: {MODEL_PATH}")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=8192,
                verbose=False)
    out = {}

    # --- 口径偏差：tools= 渲染 vs prompt_render 不传 tools ---
    messages = [
        {"role": "system",
         "content": "当前日期 2026-10-14，你可以使用 web_search 工具获取实时信息。"},
        {"role": "user", "content": search_query + " /no_think"},
    ]
    resp = llm.create_chat_completion(messages=messages, tools=TOOLS,
                                      tool_choice="auto", max_tokens=1,
                                      temperature=0.7)
    usage_prompt = resp["usage"]["prompt_tokens"]
    rendered = prompt_render.render_prompt(llm, messages)
    no_tools_tokens = prompt_render.count_text(llm, rendered)
    out["prompt_deviation"] = {
        "with_tools_prompt_tokens": usage_prompt,
        "render_prompt_no_tools_tokens": no_tools_tokens,
        "diff": usage_prompt - no_tools_tokens,
    }
    print(f"[p3] 口径偏差: tools={usage_prompt} vs render={no_tools_tokens} "
          f"(diff {usage_prompt - no_tools_tokens})")

    # --- 决策轮（真实生成，确认 tool_call 形态在 8192 ctx / CPU 下不变）---
    resp = llm.create_chat_completion(messages=messages, tools=TOOLS,
                                      tool_choice="auto", max_tokens=512,
                                      temperature=0.7)
    decision_content = resp["choices"][0]["message"]["content"] or ""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", decision_content,
                  re.DOTALL)
    out["decision_round"] = {"has_tool_call": bool(m),
                             "raw": (m.group(1) if m else "")[:300],
                             "content_snippet": decision_content[:300]}
    print(f"[p3] 决策轮 tool_call: {bool(m)}")

    # --- 回答轮：tool 角色渲染结构 + 基于真实搜索结果的回答 ---
    tool_response = "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
        for i, r in enumerate(search_results, 1))
    round2 = messages + [
        {"role": "assistant", "content": decision_content},
        {"role": "tool", "content": tool_response},
    ]
    rendered2 = prompt_render.render_prompt(llm, round2)
    tm = re.search(r"<\|im_start\|>(\w+)\n(<tool_response>.*?)</tool_response>",
                   rendered2, re.DOTALL)
    out["tool_role_render"] = {
        "tool_response_wrapped_as": tm.group(1) if tm else None,
        "block_snippet": (tm.group(0)[:300] if tm else ""),
    }
    print(f"[p3] tool 角色渲染: wrapped_as={tm.group(1) if tm else None}")

    t0 = time.perf_counter()
    resp = llm.create_chat_completion(messages=round2, max_tokens=300,
                                      temperature=0.7)
    answer = resp["choices"][0]["message"]["content"] or ""
    out["answer_round"] = {"answer": answer,
                           "elapsed_s": round(time.perf_counter() - t0, 1),
                           "usage": resp["usage"]}
    print(f"[p3] 回答轮 ({out['answer_round']['elapsed_s']}s): "
          f"{answer[:150]}")
    return out


def main():
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    print("===== 1. 360 搜索抓取 =====")
    result["search"] = part1_search()

    first_ok = next((c for c in result["search"]["cases"]
                     if c.get("ok") and c["count"] >= 2), None)
    print("\n===== 2. 正文提取（trafilatura Top-2） =====")
    if first_ok:
        full_results, _ = web_search(first_ok["query"])
        result["extract"] = part2_extract(full_results)
    else:
        result["extract"] = {"skipped": "part1 全部失败"}

    print("\n===== 3. 模型验证 =====")
    search_query = QUERIES[0]
    search_results, _ = web_search(search_query)
    result["model"] = part3_model(search_query, search_results)

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
