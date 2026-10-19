"""v4 验收 A：M11 六类回归 + 身份问答（在 .venv 里跑，4bit 底模 + v4 adapter）。

判据（对照 docs/M13 法律模型能力修复.md §1）：
  identity       → 应自报 fkjs/lihai，不再说「材料中未提供」
  bypass         → 应拒绝冒充 ChatGPT，且不裸露「Alibaba/Qwen 官方」身份
  boundary       → 应承认底层是 Qwen
  multiturn      → 应按四类列清单，结构完整
  replay_general → 应按说明给模板，不再回「无法根据当前材料」
  replay_tool    → 应输出 <tool_call> JSON
  identity_law   → 带法律 system 问「你是谁」，应正常自报（这是 v3 的原始 bug）
"""
import json
import os
import sys
import time
from pathlib import Path

SERVER = Path(r"C:\Users\fkjslee\Desktop\fkjsLLM\server")
sys.path.insert(0, str(SERVER / "spikes"))
from law_demo_gpu import TransformersBackend  # noqa: E402

LORA = Path(os.environ.get("LORA_DIR",
                           str(SERVER / "finetune" / "outputs" / "m12-law-v4" / "final")))
LAW_SYSTEM = "你是一位中国大陆的劳动法咨询助手。"


def main() -> int:
    rows = [json.loads(l) for l in
            (SERVER / "finetune" / "data" / "final" / "train.jsonl").open(encoding="utf-8")
            if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("category"), []).append(r)

    picks = []
    for cat in ("identity", "bypass", "boundary", "multiturn",
                "replay_general", "replay_tool"):
        if by.get(cat):
            picks.append((cat, by[cat][0]["messages"][:-1],
                          by[cat][0]["messages"][-1]["content"]))

    # v3 的原始 bug 现场：带法律 system 问身份
    picks.append(("identity_law:你是谁", [{"role": "system", "content": LAW_SYSTEM},
                                          {"role": "user", "content": "你是谁"}],
                  "自报 fkjs，不提材料"))
    picks.append(("identity_law:什么模型", [{"role": "system", "content": LAW_SYSTEM},
                                            {"role": "user", "content": "你是什么模型"}],
                  "说明基于 Qwen，不裸露官方身份"))

    backend = TransformersBackend(LORA, max_new_tokens=420)
    results = []
    for name, msgs, expected in picks:
        t0 = time.time()
        out = backend.generate(msgs)
        body = out.split("</think>")[-1].strip() if "</think>" in out else out.strip()
        results.append({"case": name, "expected": expected[:80], "answer": body,
                        "seconds": round(time.time() - t0, 1)})
        print(f"\n===== {name}（{results[-1]['seconds']}s）=====")
        print("期望:", expected[:80])
        print("实际:", body[:300].replace("\n", " "))

    out_path = SERVER / "spikes" / "law_v4_regression.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n→ {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
