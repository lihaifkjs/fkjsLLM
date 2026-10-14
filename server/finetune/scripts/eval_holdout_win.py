"""批次 3.5/5：留出测试集身份评测（Windows HF 版，对齐 eval_identity.py）。

加载 4-bit 底模 + LoRA adapter，在 test_holdout.jsonl 上逐条生成，
按字符串规则判分：fkjs_ok（提 fkjs）/ lihai_ok（提 lihai）/
broken（自称通义千问/Qwen/ChatGPT，边界类「承认底层」不算）。

用法: python eval_holdout_win.py --adapter outputs/mini-validate/final [--n 10]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_FINETUNE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_FINETUNE / "models" / "Qwen3-14B"
TEST_FILE = REPO_FINETUNE / "data" / "final" / "test_holdout.jsonl"
CURATED_DIR = REPO_FINETUNE / "data" / "curated"

SELF_CLAIM_RE = re.compile(r"我是(通义千问|Qwen)|我叫(通义千问|Qwen)|I('m| am) (Qwen|ChatGPT)|我是 ?ChatGPT")


def load_categories() -> dict:
    m = {}
    for p in CURATED_DIR.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            m[r["question"]] = r["category"]
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=0, help="只评前 N 条（0=全部）")
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    cat_of = load_categories()
    rows = [json.loads(l) for l in TEST_FILE.open(encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[:args.n]

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"加载完成：{args.base} + {args.adapter}，评测 {len(rows)} 条")

    stats, failures = {}, []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        q = msgs[-1]["content"]
        # 与线上 /no_think 口径一致（同 eval_identity.py）
        msgs[-1] = {**msgs[-1], "content": q + " /no_think"}
        cat = cat_of.get(q, "unknown")
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        fkjs, lihai = "fkjs" in text, "lihai" in text
        broken = bool(SELF_CLAIM_RE.search(text))
        s = stats.setdefault(cat, {"n": 0, "fkjs": 0, "lihai": 0, "broken": 0})
        s["n"] += 1; s["fkjs"] += fkjs; s["lihai"] += lihai; s["broken"] += broken
        mark = "OK " if (fkjs and not broken) else "!! "
        print(f"{mark}[{cat}] Q: {q}\n    A: {text[:120]}")
        if not fkjs or broken:
            failures.append({"cat": cat, "q": q, "got": text})

    print("\n===== 按类别统计 =====")
    for cat, s in sorted(stats.items()):
        print(f"[{cat}] n={s['n']}  提fkjs {s['fkjs']}/{s['n']}  "
              f"提lihai {s['lihai']}/{s['n']}  身份失守 {s['broken']}/{s['n']}")
    out = REPO_FINETUNE / "data" / "final" / "eval_failures_win.json"
    out.write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    print(f"失败/可疑 {len(failures)} 条 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
