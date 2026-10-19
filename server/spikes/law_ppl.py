"""测通用文本困惑度（PPL），用于判断微调是否损伤底模的语言能力。

数据：`curated/replay_general.jsonl` 里**未进入训练集**的样本（800 条抽 120 条做 replay，
剩下的留作测集）——既离线可得，又正好测「通用指令组织能力」这件事。

一个进程只加载一个模型：之前实测同一进程里 del 掉 HfBackend 再建第二个会崩
（0xC0000005），所以按 tag 分别跑，各自写 JSON，最后用 plot_law_curves.py ppl 合并画图。

用法（在 server 目录，.venv 里跑）：
  python spikes/law_ppl.py --tag base
  python spikes/law_ppl.py --tag v3 --lora finetune/outputs/m12-law-v3/final
  python spikes/law_ppl.py --tag v4 --lora finetune/outputs/m12-law-v4/final
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
CURATED = SERVER / "finetune" / "data" / "curated" / "replay_general.jsonl"
MIX_TRAIN = SERVER / "finetune" / "data" / "law" / "final_mix_v2" / "train.jsonl"
BASE_HF = SERVER / "finetune" / "models" / "Qwen3-14B"
OUT_DIR = SERVER / "spikes"
N_SAMPLES = 60
SEED = 7


def held_out_rows() -> list[dict]:
    """训练集里用到的 replay_general 答案集合，从 800 条里剔除后取样。

    curated 文件的字段是 {category, system, question, answer}，需要先转成 messages。
    """
    used = set()
    if MIX_TRAIN.exists():
        for line in MIX_TRAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") == "replay_general":
                used.add(r["messages"][-1]["content"])
    raw = [json.loads(l) for l in CURATED.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for r in raw:
        if not r.get("question") or not r.get("answer"):
            continue
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["question"]})
        msgs.append({"role": "assistant", "content": r["answer"]})
        rows.append({"messages": msgs})
    pool = [r for r in rows if r["messages"][-1]["content"] not in used]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    return pool[:N_SAMPLES]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_HF),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.lora))
    model.eval()

    rows = held_out_rows()
    total_nll, total_tokens = 0.0, 0
    for r in rows:
        msgs = r["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        # 只对 assistant 部分计入：用 labels 掩掉 prompt 段
        labels = enc["input_ids"].clone()
        prompt_len = tok(tok.apply_chat_template(msgs[:-1], tokenize=False,
                                                 add_generation_prompt=True),
                         return_tensors="pt").input_ids.shape[1]
        labels[:, :prompt_len] = -100
        with torch.inference_mode():
            out = model(**enc, labels=labels)
        n = int((labels != -100).sum())
        if n == 0:
            continue
        total_nll += float(out.loss) * n
        total_tokens += n

    ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    payload = {"tag": args.tag, "ppl": round(ppl, 3), "samples": len(rows),
               "tokens": total_tokens, "lora": args.lora}
    out = OUT_DIR / f"law_ppl_{args.tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag}: PPL={ppl:.3f}（{len(rows)} 条 / {total_tokens} tokens）→ {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
