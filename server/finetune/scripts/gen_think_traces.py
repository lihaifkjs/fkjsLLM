"""M11：用 base GGUF 给通用语料生成 think 轨迹。

把 single-turn 语料的 answer 从「直答」改写为
`<think>\\n思考\\n</think>\\n\\n正文`，让 SFT 重新学会思考（背景见 docs/M11 SFT再训练.md）。
**身份三类（identity/bypass/boundary）与 replay_tool 不要用本脚本**——前者的身份口径
是直答，后者要保持 /no_think 的形态可判别性（M9 决策轮依赖）。

三个实现要点：
1. 思考与正文都用 base 模型自己生成的（自洽）；不用它的思考配别人的答案
2. 已含 `</think>` 的条目默认跳过 → 800 条长跑可中断续跑，重跑不会重复生成
3. 未闭合 / 思考为空 / 超过 --max-think-chars 的条目**不采用**（保留原直答并计入
   统计）——宁可少几条 thinking，也不把半截推理或超长样本塞进 loss

`--system-hint`（默认开启）只在**生成时**给模型一个 system 消息要求思考简洁；
它不会写进样本（样本仍只有 user + assistant），所以不引入 train/serve 偏斜。
实测：不加提示时 base 的思考中位 695 字，会让样本超出 train_windows 的 max_seq。

用法（需要带 llama-cpp-python 的环境：Mac 是 server/.venv，Windows 是服务侧环境）:
  # 本机冒烟（3 条，写到临时目录）
  ../.venv/bin/python3 gen_think_traces.py --input data/curated/replay_general.jsonl \
      --output data/smoke/replay_general.jsonl --limit 3
  # Windows 全量（原地改写；data/ 已入 git，改坏了 git checkout 还原）
  python gen_think_traces.py --input data/curated/replay_general.jsonl \
      --output data/curated/replay_general.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from llama_cpp import Llama, llama_supports_gpu_offload

REPO = Path(__file__).resolve().parents[3]  # 仓库根目录（GGUF 所在处）
DEFAULT_MODEL = REPO / "Qwen3-14B-Q4_K_M.gguf"  # base 原版，默认就带 think
# 生成期专用：压短思考，避免样本超出 max_seq（不写进样本）
DEFAULT_HINT = ("回答前先在思考里做简短推理（120 字以内）：只写关键判断，"
                "不要复述题目，不要展开无关分支，然后给出答案。")


def split_output(text: str) -> tuple[str, str] | None:
    """模型输出 → (思考, 正文)；未闭合或无思考返回 None。"""
    if "</think>" not in text:
        return None
    head, body = text.split("</think>", 1)
    think = head.split("<think>")[-1].strip()
    if not think:
        return None
    return think, body.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True,
                    help="可与 --input 相同（原地改写）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-think-chars", type=int, default=500,
                    help="思考段超长则弃用该条（约束训练样本总长）")
    ap.add_argument("--system-hint", default=DEFAULT_HINT,
                    help="生成期专用 system 提示（压短思考）；传空串关闭")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计待处理条数，不加载模型")
    args = ap.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    entries = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    todo = [i for i, e in enumerate(entries) if "</think>" not in e["answer"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{in_path.name}: 共 {len(entries)} 条，待生成 {len(todo)} 条"
          f"（已含 think 跳过 {len(entries) - len(todo)} 条）")
    if args.dry_run or not todo:
        return 0

    llm = Llama(model_path=args.model,
                n_ctx=8192,
                n_gpu_layers=-1 if llama_supports_gpu_offload() else 0,
                verbose=False)
    stats, rejected = [], []  # 成功/未采用的思考长度，用于定 --max-think-chars
    failed = {"未思考（直答）": 0, "think 未闭合（疑似超长截断）": 0,
              "思考为空": 0, "思考过长": 0}
    t0 = time.perf_counter()
    for n, idx in enumerate(todo, 1):
        entry = entries[idx]
        messages = ([{"role": "system", "content": args.system_hint}]
                    if args.system_hint else [])
        messages.append({"role": "user", "content": entry["question"]})
        out = llm.create_chat_completion(
            messages=messages, temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=args.max_tokens, stream=False)
        text = out["choices"][0]["message"]["content"] or ""
        parsed = split_output(text)
        if parsed is None:
            if "<think>" not in text:
                failed["未思考（直答）"] += 1
            elif "</think>" not in text:
                failed["think 未闭合（疑似超长截断）"] += 1
            else:
                failed["思考为空"] += 1
        else:
            think, body = parsed
            if len(think) > args.max_think_chars:
                failed["思考过长"] += 1
                rejected.append(len(think))
            elif not body:
                failed["思考为空"] += 1
            else:
                entry["answer"] = f"<think>\n{think}\n</think>\n\n{body}"
                stats.append(len(think))
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] 成功 {len(stats)} 失败 "
                  f"{sum(failed.values())} | {time.perf_counter() - t0:.0f}s",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n生成 {len(stats)} 条 → {out_path}")
    if stats:
        stats.sort()
        print(f"思考长度 中位 {statistics.median(stats):.0f} / p95 "
              f"{stats[int(len(stats) * 0.95)]} / max {stats[-1]} 字")
    if rejected:
        rejected.sort()
        print(f"过长未采用 {len(rejected)} 条：中位 "
              f"{statistics.median(rejected):.0f} / max {rejected[-1]} 字"
              f"（当前上限 {args.max_think_chars}，可 --max-think-chars 放宽，"
              f"但需同步确认 train_windows 的 --max-seq）")
    print(f"未采用：{failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
