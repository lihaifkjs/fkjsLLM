"""批次 1（M11 修订）：渲染口径校验。

用模型目录自带的 Qwen3 chat template 渲染 data/final/*.jsonl，验证：
  1. 模板渲染正常，无特殊 token 注入（<|im_start|> 等不应出现在 content 内部）
  2. think 口径**按类别判定**（M11 决策 B）：
     - identity / bypass / boundary / replay_tool：语料不得含非空 think 段
       （模板会给最后一条 assistant 注入空壳 `<think>\\n\\n</think>`，属预期行为）
     - replay_general：带 think 的比例不低于 --min-think-ratio（生成时「思考过长/
       未闭合」的条目会保留直答，所以看覆盖率而不是硬要求）
     - multiturn：至少一条 assistant 有非空 think（末轮可以是直答的身份轮）
     硬失败：任何「有 <think> 未闭合」或「声明了却为空」（M8 就是把空壳训进了 loss）；
     覆盖率不足默认只告警，加 --strict 才计入失败
  3. 复现 train_windows.pretokenize 的掩码口径：prompt 是 full 的前缀、监督区间从
     prompt 之后开始；并断言 **prompt 里不出现 <think>**（证明历史轮 think 被模板
     剥掉、不会污染上下文）
  4. token 长度分布（--max-seq 是 train_windows 的丢弃阈值）

用法: python render_check.py [--model server/finetune/models/Qwen3-14B-4bit]
                            [--data-dir server/finetune/data/final] [--max-seq 640]
                            [--min-think-ratio 0.5] [--strict]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO_FINETUNE = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_FINETUNE / "data" / "final"
DEFAULT_MODEL = REPO_FINETUNE / "models" / "Qwen3-14B-4bit"

SPECIAL_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
# none=不得有非空 think；ratio=覆盖率达标（--min-think-ratio）；some=至少一轮有
THINK_POLICY = {
    "identity": "none",
    "bypass": "none",
    "boundary": "none",
    "replay_tool": "none",
    "replay_general": "ratio",
    "multiturn": "some",
}


def think_text(content: str) -> tuple[str, bool]:
    """→ (think 内容, 是否畸形)。畸形＝有 <think> 无 </think>，或声明了却为空。"""
    if "<think>" not in content:
        return "", False
    if "</think>" not in content:
        return "", True
    return content.split("</think>")[0].split("<think>")[-1].strip(), False


def check_row(tokenizer, row: dict, max_seq: int) -> tuple[list[str], dict]:
    """校验单行，返回 (硬失败列表, 统计)。校验口径见模块 docstring。"""
    msgs, category = row["messages"], row.get("category", "?")
    bad: list[str] = []
    for m in msgs:
        if SPECIAL_RE.search(m["content"]):
            bad.append(f"特殊 token 注入: {m['content'][:40]!r}")

    assistants = [m for m in msgs if m["role"] == "assistant"]
    thinks, malformed = [], 0
    for m in assistants:
        text, is_bad = think_text(m["content"])
        malformed += is_bad
        thinks.append(text)
    if malformed:
        bad.append(f"{malformed} 条 assistant 的 think 段畸形（未闭合或为空）")
    has_think = any(thinks)
    policy = THINK_POLICY.get(category)
    if policy is None:
        bad.append(f"缺 category 字段或类别未知（{category}）——"
                   f"M8 旧格式的 final 文件，请先重跑 build_dataset.py")
    elif policy == "none" and has_think:
        bad.append("该类语料不得含非空 think 段")

    full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True)
    if "<think>" in prompt_text:
        bad.append("prompt 里出现了 <think>（历史轮 think 未被剥掉）")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[:len(prompt_ids)] != prompt_ids:
        bad.append("prompt 不是 full 的前缀（pretokenize 会丢弃该样本）")
        supervised = ""
    else:
        supervised = tokenizer.decode(full_ids[len(prompt_ids):])
    if thinks and not supervised.startswith("<think>"):
        bad.append("末轮有 think，但监督区间不是从 <think> 开始")

    return bad, {"category": category, "tokens": len(full_ids), "policy": policy,
                 "has_think": has_think, "over": len(full_ids) > max_seq,
                 "supervised": supervised, "full": full_text}


def check_file(tokenizer, path: Path, args, show: int = 2) -> tuple[int, list[str]]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    lens, over, fails = [], 0, 0
    per_cat: Counter = Counter()
    think_rows: Counter = Counter()
    warns: list[str] = []
    shown = 0
    for r in rows:
        bad, st = check_row(tokenizer, r, args.max_seq)
        lens.append(st["tokens"])
        over += st["over"]
        per_cat[st["category"]] += 1
        think_rows[st["category"]] += st["has_think"]
        for b in bad:
            fails += 1
            print(f"  !! {b} | {st['category']} | {st['full'][:60]!r}")
        if shown < show and (st["category"] == "multiturn" or st["has_think"]):
            shown += 1
            print(f"  ---- 渲染样例（{st['category']}，监督区间 "
                  f"{len(st['supervised'])} 字）----")
            print("  " + st["supervised"][:300].replace("\n", "\n  "))
    lens.sort()
    n = len(lens)
    print(f"[{path.name}] {n} 条 | token 中位 {lens[n // 2]} / p95 "
          f"{lens[int(n * 0.95)]} / max {lens[-1]} | >{args.max_seq}: {over} 条 | "
          f"异常 {fails}")
    for cat, cnt in sorted(per_cat.items()):
        ratio = think_rows[cat] / cnt
        flag = ""
        if THINK_POLICY.get(cat) == "ratio" and ratio < args.min_think_ratio:
            flag = f"  ⚠ 覆盖率 < {args.min_think_ratio}"
            warns.append(f"{path.name} {cat} think 覆盖率 {ratio:.0%} < "
                         f"{args.min_think_ratio:.0%}（先跑 gen_think_traces.py，"
                         f"或用 --min-think-ratio 放宽）")
        print(f"    {cat:15s} {cnt:5d} 条 | 带 think {think_rows[cat]:5d}"
              f"（{ratio:.0%}）{flag}")
    for cat, cnt in per_cat.items():
        if THINK_POLICY.get(cat) == "some" and think_rows[cat] == 0:
            warns.append(f"{path.name} {cat} 没有任何一轮带 think")
    return fails, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data-dir", default=str(FINAL_DIR))
    ap.add_argument("--max-seq", type=int, default=640,
                    help="train_windows 的超长丢弃阈值，超出的样本会被丢掉")
    ap.add_argument("--min-think-ratio", type=float, default=0.5,
                    help="replay_general 的 think 覆盖率下限（默认 50%%）")
    ap.add_argument("--strict", action="store_true",
                    help="覆盖率不足也判为失败（全量生成完后的正式校验用）")
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fails, warns = 0, []
    for name in ("train.jsonl", "valid.jsonl", "test_holdout.jsonl"):
        path = Path(args.data_dir) / name
        if path.exists():
            f, w = check_file(tokenizer, path, args)
            fails += f
            warns += w
    for w in warns:
        print(f"⚠ {w}")
    if fails:
        print(f"\n渲染校验失败：{fails} 处异常")
    elif warns and args.strict:
        print(f"\n渲染校验失败：{len(warns)} 处覆盖率不足（--strict）")
        fails = len(warns)
    else:
        print("\n渲染校验通过（覆盖率告警见上）" if warns else "\n渲染校验通过")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    sys.exit(main())
