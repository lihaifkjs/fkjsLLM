"""批次 1：渲染口径校验。

用模型目录自带的 Qwen3 chat template 渲染 data/final/*.jsonl，验证：
  1. 模板渲染正常，无异常字符注入（<|im_start|> 等不应出现在 content 内部）
  2. 训练目标不含 think 内容（与线上 /no_think 口径一致；空 think 块为模板行为，不算异常）
  3. token 长度分布（Windows 训练 max seq 512-768，超长样本需回到数据侧处理）

用法: .../python render_check.py [--model server/finetune/models/Qwen3-14B-4bit]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_FINETUNE = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_FINETUNE / "data" / "final"
DEFAULT_MODEL = REPO_FINETUNE / "models" / "Qwen3-14B-4bit"

SPECIAL_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
NONEMPTY_THINK_RE = re.compile(r"<think>\s*[^<\s]")


def check_file(tokenizer, path: Path, show: int = 2) -> bool:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    lens, bad = [], 0
    for r in rows:
        msgs = r["messages"]
        for m in msgs:
            if SPECIAL_RE.search(m["content"]):
                print(f"  !! 特殊 token 注入: {m['content'][:60]!r}")
                bad += 1
        if NONEMPTY_THINK_RE.search(msgs[-1]["content"]):
            print(f"  !! assistant 含 think 内容: {msgs[-1]['content'][:60]!r}")
            bad += 1
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        lens.append(len(tokenizer(text).input_ids))
    lens.sort()
    n = len(lens)
    over = sum(1 for l in lens if l > 768)
    print(f"[{path.name}] {n} 条 | token min/中位/p95/max: "
          f"{lens[0]}/{lens[n // 2]}/{lens[int(n * 0.95)]}/{lens[-1]} | >768: {over} 条 | 异常: {bad}")
    for r in rows[:show]:
        print("  ---- 渲染样例 ----")
        print("  " + tokenizer.apply_chat_template(r["messages"], tokenize=False).replace("\n", "\n  "))
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ok = True
    for name in ("train.jsonl", "valid.jsonl", "test_holdout.jsonl"):
        path = FINAL_DIR / name
        if path.exists():
            ok &= check_file(tokenizer, path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
