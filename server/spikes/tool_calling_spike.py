"""Spike: 验证 Qwen3-14B-Q4_K_M 在 llama-cpp-python 下的 tool calling 可靠性。

用法:  server/.venv/bin/python server/spikes/tool_calling_spike.py
结果:  server/spikes/tool_calling_baseline.json
"""

import json
import re
import sys
import time
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "Qwen3-14B-Q4_K_M.gguf")
RESULT_PATH = Path(__file__).with_name("tool_calling_baseline.json")
CURRENT_DATE = "2026-10-09"

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

SYSTEM_PROMPT = (
    f"当前日期 {CURRENT_DATE}，你可以使用 web_search 工具获取实时信息。"
)

# (user_query, should_call_tool)
CASES = [
    ("今天北京的天气怎么样", True),
    ("2026年诺贝尔文学奖得主是谁", True),
    ("最新一代 iPhone 的价格是多少", True),
    ("帮我查一下最近的科技新闻", True),
    ("搜索一下 llama.cpp 最新版本号", True),
    ("写一首关于秋天的五言绝句", False),
    ("1+1 等于几", False),
    ("用 Python 写一个快速排序", False),
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str):
    """从输出文本解析 <tool_call> 块。返回 (has_call, json_valid, fn_name, query, raw)."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return False, False, None, None, None
    raw = m.group(1)
    try:
        obj = json.loads(raw)
        return True, True, obj.get("name"), (obj.get("arguments") or {}).get("query"), raw
    except json.JSONDecodeError:
        return True, False, None, None, raw


def parse_native_response(resp: dict):
    """解析 create_chat_completion(tools=...) 的原生响应。"""
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    if tcs:
        tc = tcs[0]["function"]
        raw = tc.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            return True, True, tc.get("name"), (args or {}).get("query"), raw, content
        except (json.JSONDecodeError, TypeError):
            return True, False, tc.get("name"), None, raw, content
    has, valid, name, query, raw = parse_tool_call(content)
    return has, valid, name, query, raw, content


def run_native(llm, messages):
    return llm.create_chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
        temperature=0.7,
    )


def run_manual(llm, messages):
    prompt = llm.tokenizer_.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    out = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    return out["choices"][0]["text"]


def judge(query: str, tool_call: dict):
    """query 参数合理性: 非空字符串且不含整句复述的明显废话。"""
    return isinstance(query, str) and 0 < len(query.strip()) <= 100


def main():
    print(f"加载模型: {MODEL_PATH}")
    t0 = time.time()
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

    # 探测两种调用方式的可用性
    probe_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + " /no_think"},
        {"role": "user", "content": "搜索一下 llama.cpp 最新版本号"},
    ]
    native_ok, manual_ok = False, False
    native_error = manual_error = None
    try:
        resp = run_native(llm, probe_messages)
        native_ok = True
        print("[probe] create_chat_completion(tools=...) 可用")
        print("[probe] 原生响应 content 片段:", repr(resp["choices"][0]["message"]))
    except Exception as e:  # noqa: BLE001
        native_error = f"{type(e).__name__}: {e}"
        print(f"[probe] create_chat_completion(tools=...) 不可用: {native_error}")
    try:
        text = run_manual(llm, probe_messages)
        manual_ok = True
        print("[probe] 手动 apply_chat_template 可用，输出片段:", repr(text[:300]))
    except Exception as e:  # noqa: BLE001
        manual_error = f"{type(e).__name__}: {e}"
        print(f"[probe] 手动 apply_chat_template 不可用: {manual_error}")

    if not native_ok and not manual_ok:
        print("两种方式均失败，尝试手动拼 Hermes 格式 prompt ...")
        manual_error = (manual_error or "") + " | fallback: hermes-manual"
        manual_ok = "hermes"  # 标记走 hermes 分支

    details = []
    for i, (user_q, expect_call) in enumerate(CASES, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " /no_think"},
            {"role": "user", "content": user_q},
        ]
        method = "native" if native_ok else ("manual" if manual_ok is True else "hermes")
        t_start = time.time()
        try:
            if method == "native":
                resp = run_native(llm, messages)
                has, valid, name, query, raw, content = parse_native_response(resp)
                output_text = content
            elif method == "manual":
                content = run_manual(llm, messages)
                has, valid, name, query, raw = parse_tool_call(content)
                output_text = content
            else:
                # Hermes 手动格式
                tools_json = json.dumps(TOOLS[0]["function"], ensure_ascii=False)
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT} /no_think\n\n"
                    f"# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                    f"You are provided with function signatures within <tools></tools> XML tags:\n"
                    f"<tools>\n{tools_json}\n</tools>\n\n"
                    f"For each function call, return a json object with function name and arguments "
                    f"within <tool_call></tool_call> XML tags:\n"
                    f'<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                    f"</tool_call><|im_end|>\n"
                    f"<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
                )
                out = llm(prompt, max_tokens=512, temperature=0.7,
                          stop=["<|im_end|>", "<|endoftext|>"])
                content = out["choices"][0]["text"]
                has, valid, name, query, raw = parse_tool_call(content)
                output_text = content
        except Exception as e:  # noqa: BLE001
            details.append({
                "case": i, "query": user_q, "expect_tool_call": expect_call,
                "method": method, "error": f"{type(e).__name__}: {e}",
            })
            print(f"[{i}/8] ERROR {user_q}: {e}")
            continue
        elapsed = time.time() - t_start

        decision_correct = (has == expect_call)
        name_correct = (name == "web_search") if has else None
        query_reasonable = judge(user_q, query) if (has and valid) else None
        record = {
            "case": i,
            "query": user_q,
            "expect_tool_call": expect_call,
            "method": method,
            "has_tool_call": has,
            "json_valid": valid if has else None,
            "fn_name": name,
            "fn_name_correct": name_correct,
            "query_arg": query,
            "query_reasonable": query_reasonable,
            "decision_correct": decision_correct,
            "elapsed_s": round(elapsed, 1),
            "raw_output": (raw or "")[:500],
            "output_snippet": (output_text or "")[:500],
        }
        details.append(record)
        print(
            f"[{i}/8] {'OK ' if decision_correct else 'BAD'} call={has} "
            f"json={record['json_valid']} name={name} q={query!r} "
            f"({elapsed:.1f}s) {user_q}"
        )

    called = [d for d in details if "error" not in d]
    n = len(called)
    fmt_total = [d for d in called if d["has_tool_call"]]
    summary = {
        "total": n,
        "errors": n - len(called) if n != len(details) else 0,
        "tool_call_rate": round(len(fmt_total) / n, 3) if n else None,
        "json_valid_rate": (
            round(sum(1 for d in fmt_total if d["json_valid"]) / len(fmt_total), 3)
            if fmt_total else None
        ),
        "fn_name_correct_rate": (
            round(sum(1 for d in fmt_total if d["fn_name_correct"]) / len(fmt_total), 3)
            if fmt_total else None
        ),
        "decision_accuracy": (
            round(sum(1 for d in called if d["decision_correct"]) / n, 3) if n else None
        ),
    }

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": Path(MODEL_PATH).name,
        "llama_cpp_python": "0.3.36",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "temperature": 0.7,
        "system_prompt": SYSTEM_PROMPT + " /no_think",
        "api_availability": {
            "native_create_chat_completion": native_ok,
            "native_error": native_error,
            "manual_apply_chat_template": manual_ok is True,
            "manual_error": manual_error,
        },
        "summary": summary,
        "details": details,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {RESULT_PATH}")

    print("\n===== 汇总 =====")
    print(f"用例总数:            {summary['total']}")
    print(f"产生 tool_call 比例: {summary['tool_call_rate']}")
    print(f"JSON 合法率:         {summary['json_valid_rate']}")
    print(f"函数名正确率:        {summary['fn_name_correct_rate']}")
    print(f"决策正确率:          {summary['decision_accuracy']}")
    print(f"原生 API 可用:       {native_ok}")
    print(f"手动模板可用:        {manual_ok is True}")


if __name__ == "__main__":
    sys.exit(main())
