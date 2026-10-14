"""M8 验收：fkjs 微调模型 tool calling 回归对拍。

用例与判分和 spikes/tool_calling_spike.py 完全一致（同 web_search 工具定义、
同 system prompt、同 8 条用例），模型换成 fkjs-14b-q4_k_m.gguf，
结果与 spikes/tool_calling_baseline.json（决策 87.5% / 格式 100%）对比。

用法: server/.venv/bin/python server/finetune/scripts/tool_calling_regression.py
"""

import json
import re
import sys
import time
from pathlib import Path

from llama_cpp import Llama

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(REPO_ROOT / "fkjs-14b-q4_k_m.gguf")
BASELINE_PATH = REPO_ROOT / "server" / "spikes" / "tool_calling_baseline.json"
RESULT_PATH = REPO_ROOT / "server" / "finetune" / "tool_calling_regression.json"

# 与 spike 完全一致的契约
SYSTEM_PROMPT = "当前日期 2026-10-14，你可以使用 web_search 工具获取实时信息。 /no_think"
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


def main():
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=0, verbose=False)
    details = []
    for i, (q, expect_call) in enumerate(CASES, 1):
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": q}],
            tools=TOOLS, tool_choice="auto", max_tokens=512, temperature=0.7)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []
        has = valid = False
        name = query = None
        if tcs:
            fn = tcs[0]["function"]
            has = True
            try:
                args = json.loads(fn.get("arguments") or "{}")
                valid, name, query = True, fn.get("name"), args.get("query")
            except json.JSONDecodeError:
                name = fn.get("name")
        else:
            m = TOOL_CALL_RE.search(content)
            if m:
                has = True
                try:
                    obj = json.loads(m.group(1))
                    valid, name, query = True, obj.get("name"), (obj.get("arguments") or {}).get("query")
                except json.JSONDecodeError:
                    pass
        correct = has == expect_call
        details.append({"case": i, "query": q, "expect": expect_call, "has_call": has,
                        "json_valid": valid if has else None, "fn_name": name,
                        "correct": correct, "elapsed_s": round(time.time() - t0, 1)})
        print(f"[{i}/8] {'OK ' if correct else 'BAD'} call={has} name={name} ({time.time()-t0:.0f}s) {q}", flush=True)

    n = len(details)
    calls = [d for d in details if d["has_call"]]
    result = {
        "model": "fkjs-14b-q4_k_m.gguf",
        "decision_accuracy": sum(d["correct"] for d in details) / n,
        "json_valid_rate": (sum(1 for d in calls if d["json_valid"]) / len(calls)) if calls else None,
        "fn_name_correct_rate": (sum(1 for d in calls if d["fn_name"] == "web_search") / len(calls)) if calls else None,
        "baseline": {"decision_accuracy": 0.875, "json_valid_rate": 1.0},
        "details": details,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n决策正确率: {result['decision_accuracy']:.3f}（基线 0.875）")
    print(f"JSON 合法率: {result['json_valid_rate']}（基线 1.0）")
    print(f"结果 → {RESULT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
