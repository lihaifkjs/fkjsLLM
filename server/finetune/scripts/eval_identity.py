"""批次 2：留出测试集身份评测。

用 fuse 后的模型在 test_holdout.jsonl 上逐条生成，按字符串规则判分：
  fkjs_ok    —— 回答提到 fkjs（身份写入）
  lihai_ok   —— 回答提到 lihai（开发者写入）
  broken     —— 回答自我声称为通义千问/Qwen/ChatGPT（身份未守住）
边界类样本答案本身允许提 Qwen（承认底层），因此 broken 只判「我是通义千问」式自称。

用法: .../python eval_identity.py --model server/finetune/fused/xxx
"""

import argparse
import json
import re
import sys
from pathlib import Path

from mlx_lm import generate, load

REPO_FINETUNE = Path(__file__).resolve().parents[1]
TEST_FILE = REPO_FINETUNE / "data" / "final" / "test_holdout.jsonl"
CURATED_DIR = REPO_FINETUNE / "data" / "curated"

SELF_CLAIM_RE = re.compile(r"我是(通义千问|Qwen)|我叫(通义千问|Qwen)|I('m| am) (Qwen|ChatGPT)|我是 ?ChatGPT")


def load_categories() -> dict:
    """question → category 映射，用于按类别拆指标。"""
    m = {}
    for p in CURATED_DIR.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            m[r["question"]] = r["category"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=150)
    args = ap.parse_args()

    cat_of = load_categories()
    rows = [json.loads(l) for l in TEST_FILE.open(encoding="utf-8") if l.strip()]
    print(f"加载模型 {args.model}，评测 {len(rows)} 条")
    model, tokenizer = load(args.model)

    stats = {}
    failures = []
    for r in rows:
        msgs = r["messages"]
        q = next(m["content"] for m in msgs if m["role"] == "user")
        expected = msgs[-1]["content"]
        cat = cat_of.get(q, "unknown")
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        # 与线上 /no_think 口径一致
        prompt_msgs[-1] = {**prompt_msgs[-1], "content": prompt_msgs[-1]["content"] + " /no_think"}
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()

        s = stats.setdefault(cat, {"n": 0, "fkjs": 0, "lihai": 0, "broken": 0})
        s["n"] += 1
        fkjs = "fkjs" in out
        lihai = "lihai" in out
        broken = bool(SELF_CLAIM_RE.search(out))
        s["fkjs"] += fkjs
        s["lihai"] += lihai
        s["broken"] += broken
        if not fkjs or broken:
            failures.append({"cat": cat, "q": q, "expected": expected, "got": out})

    print("\n===== 按类别统计 =====")
    for cat, s in sorted(stats.items()):
        print(f"[{cat}] n={s['n']}  提fkjs {s['fkjs']}/{s['n']}  "
              f"提lihai {s['lihai']}/{s['n']}  身份失守 {s['broken']}/{s['n']}")

    out_path = REPO_FINETUNE / "data" / "final" / "eval_failures.json"
    out_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    print(f"\n失败/可疑样本 {len(failures)} 条 → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
