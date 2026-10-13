"""批次 1：数据集组装。

把 data/curated/*.jsonl（人工筛改后的定稿）按目标配比合并，
身份三类留 20% 进留出测试集，replay 全进训练集；
输出 mlx-lm / HF 通用的 {"messages": [...]} JSONL：
  data/final/train.jsonl / valid.jsonl / test_holdout.jsonl

用法: .../python build_dataset.py [--input-dir data/curated] [--counts identity=200,...]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_FINETUNE = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_FINETUNE / "data" / "final"

# 目标配比（M8 计划 §4.1）；curated 不足时取全部并警告
DEFAULT_COUNTS = {
    "identity": 200,
    "bypass": 60,
    "boundary": 40,
    "replay_general": 700,
    "replay_tool": 150,
}
HOLDOUT_CATEGORIES = {"identity", "bypass", "boundary"}
HOLDOUT_RATIO = 0.2


def to_messages(entry: dict) -> dict:
    msgs = []
    if entry.get("system"):
        msgs.append({"role": "system", "content": entry["system"]})
    msgs.append({"role": "user", "content": entry["question"]})
    msgs.append({"role": "assistant", "content": entry["answer"]})
    return {"messages": msgs}


def load_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(REPO_FINETUNE / "data" / "curated"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    in_dir = Path(args.input_dir)

    train, test = [], []
    for category, target in DEFAULT_COUNTS.items():
        path = in_dir / f"{category}.jsonl"
        if not path.exists():
            print(f"[{category}] 缺文件 {path}，跳过")
            continue
        entries = load_entries(path)
        rng.shuffle(entries)
        if len(entries) < target:
            print(f"[{category}] 仅有 {len(entries)} 条（目标 {target}），全部取用")
        picked = entries[:target]
        if category in HOLDOUT_CATEGORIES:
            n_test = max(1, round(len(picked) * HOLDOUT_RATIO))
            test += picked[:n_test]
            train += picked[n_test:]
        else:
            train += picked
        print(f"[{category}] 取 {len(picked)} 条（train {len(picked) - (n_test if category in HOLDOUT_CATEGORIES else 0)}"
              f"{f' / test {n_test}' if category in HOLDOUT_CATEGORIES else ''}）")

    rng.shuffle(train)
    rng.shuffle(test)
    n_valid = max(20, round(len(train) * 0.05))
    valid, train = train[:n_valid], train[n_valid:]

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(FINAL_DIR / "train.jsonl", [to_messages(e) for e in train])
    write_jsonl(FINAL_DIR / "valid.jsonl", [to_messages(e) for e in valid])
    write_jsonl(FINAL_DIR / "test_holdout.jsonl", [to_messages(e) for e in test])
    print(f"\ntrain {len(train)} / valid {len(valid)} / test_holdout {len(test)} → {FINAL_DIR}")


if __name__ == "__main__":
    sys.exit(main())
