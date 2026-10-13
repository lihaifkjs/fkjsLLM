"""批次 1：身份/绕过/边界候选样本生成。

种子问题经 MLX 模型改写扩充口吻变体，答案原样保留（identity_config 定稿），
产出 data/raw/{identity,bypass,boundary}.jsonl 供人工筛改（→ data/curated/）。

用法: server/finetune/.venv/bin/python server/finetune/scripts/gen_candidates.py
"""

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).parent))
from identity_config import (  # noqa: E402
    BYPASS_SEEDS,
    IDENTITY_ANSWERS,
    IDENTITY_QUESTIONS,
    BOUNDARY_SEEDS,
)

REPO_FINETUNE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_FINETUNE / "models" / "Qwen3-14B-4bit"
RAW_DIR = REPO_FINETUNE / "data" / "raw"

PARAPHRASE_PROMPT = (
    "把下面这个向 AI 助手提的问题改写成 {n} 种不同问法。"
    "要求：意思不变；口吻多样（口语、正式、随意、英文均可）；一行一个；"
    "不要编号、不要解释、不要回答问题本身。\n原问题：{q}\n/no_think"
)


def paraphrase(model, tokenizer, sampler, question: str, n: int) -> list[str]:
    messages = [{"role": "user", "content": PARAPHRASE_PROMPT.format(n=n, q=question)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = generate(
        model, tokenizer, prompt=prompt, max_tokens=300, sampler=sampler, verbose=False
    )
    variants = []
    for line in out.splitlines():
        # 去掉模型可能加的序号/项目符号前缀；过滤 think 块等模板噪声行
        v = line.strip().lstrip("0123456789.、-·* ").strip().strip('"“”')
        if v.startswith("<") or "think>" in v:
            continue
        if 4 <= len(v) <= 120 and v != question:
            variants.append(v)
    return variants


def emit(records: list[dict], category: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{category}.jsonl"
    seen = set()
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            key = r["question"].casefold()
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{category}] 写入 {n} 条 → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=4, help="每个种子改写出的变体数")
    args = ap.parse_args()

    print(f"加载模型: {args.model}")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.9, top_p=0.95)

    buckets = {
        # identity：答案在模板间轮转
        "identity": [
            (q, IDENTITY_ANSWERS[i % len(IDENTITY_ANSWERS)])
            for i, q in enumerate(IDENTITY_QUESTIONS)
        ],
        "bypass": BYPASS_SEEDS,
        "boundary": BOUNDARY_SEEDS,
    }
    for category, seeds in buckets.items():
        records = []
        for q, a in seeds:
            records.append({
                "category": category, "system": None,
                "question": q, "answer": a, "source": "seed", "seed": q,
            })
            for v in paraphrase(model, tokenizer, sampler, q, args.n):
                records.append({
                    "category": category, "system": None,
                    "question": v, "answer": a, "source": "paraphrase", "seed": q,
                })
            print(f"  [{category}] 种子「{q[:20]}」→ {len(records) - 1} 条累计")
        emit(records, category)


if __name__ == "__main__":
    sys.exit(main())
