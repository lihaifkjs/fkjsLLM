"""批次 1（M11 修订）：数据集组装。

把 data/curated/*.jsonl（人工定稿 + 生成）按目标配比合并，身份类与多轮类各留
20% 进留出测试集，replay 全进训练集；输出 HF/MLX 通用的
`{"category": ..., "messages": [...]}` JSONL：final/train.jsonl、valid.jsonl、
test_holdout.jsonl。

两种输入结构：
- 单轮 `{question, answer}`（identity / bypass / boundary / replay_general / replay_tool）
- 多轮 `{messages: [{role, content}...]}`（multiturn，M11 新增）

**前缀展开（M11 关键变化）**：N 轮会话 → N 个样本（每个 assistant 轮各当一次
「最后一条」被监督）。模板会剥掉历史轮的 think，不展开则前面几轮的思考完全进不了
loss（决策 D，见 docs/M11 SFT再训练.md §3）。展开放在**切分之后**，保证同一会话的
所有前缀落在同一个 split，不泄题；留出集不展开（评估用整段会话）。

用法: python build_dataset.py [--input-dir data/curated] [--output-dir data/final]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_FINETUNE = Path(__file__).resolve().parents[1]
CURATED_DIR = REPO_FINETUNE / "data" / "curated"
FINAL_DIR = REPO_FINETUNE / "data" / "final"

# 目标配比（M8 计划 §4.1 + M11 多轮；curated 不足时取全部并警告）
DEFAULT_COUNTS = {
    "identity": 200,
    "bypass": 60,
    "boundary": 40,
    "replay_general": 700,
    "replay_tool": 150,
    "multiturn": 300,
}
HOLDOUT_CATEGORIES = {"identity", "bypass", "boundary"}
# 多轮不进留出集：test_holdout.jsonl 是 eval_holdout_win.py 的规则判分输入
# （身份规则），多轮样本混进去会被误判为失败；多轮能力改用多轮探针验收
HOLDOUT_RATIO = 0.2


def to_messages(entry: dict) -> list[dict]:
    """curated 条目 → messages 列表（多轮条目直接透传）。"""
    if "messages" in entry:
        return [dict(m) for m in entry["messages"]]
    msgs = []
    if entry.get("system"):
        msgs.append({"role": "system", "content": entry["system"]})
    msgs.append({"role": "user", "content": entry["question"]})
    msgs.append({"role": "assistant", "content": entry["answer"]})
    return msgs


def expand_prefixes(msgs: list[dict]) -> list[list[dict]]:
    """前缀展开：每个以 assistant 收尾的前缀 = 一个训练样本。

    单轮条目展开后仍是 1 条（等价于 M8 的旧行为）；N 轮会话展开成 N 条。
    """
    return [msgs[:i + 1] for i, m in enumerate(msgs) if m["role"] == "assistant"]


def load_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def expand_rows(rows: list[tuple[str, list[dict]]]) -> list[dict]:
    out = []
    for category, msgs in rows:
        for prefix in expand_prefixes(msgs):
            out.append({"category": category, "messages": prefix})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(CURATED_DIR))
    ap.add_argument("--output-dir", default=str(FINAL_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-expand", action="store_true",
                    help="关闭前缀展开（对照用，多轮只监督末轮）")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)

    train, test = [], []  # [(category, messages)]
    for category, target in DEFAULT_COUNTS.items():
        path = in_dir / f"{category}.jsonl"
        if not path.exists():
            print(f"[{category}] 缺文件 {path}，跳过")
            continue
        entries = load_entries(path)
        # 先按内容排序再 shuffle：抽样结果只取决于「集合内容」，
        # 与 curated 文件的行顺序无关（否则改一次行序就换了留出集）
        entries.sort(key=lambda e: e.get("question") or e["messages"][0]["content"])
        rng.shuffle(entries)
        if len(entries) < target:
            print(f"[{category}] 仅有 {len(entries)} 条（目标 {target}），全部取用")
        picked = entries[:target]
        msgs = [(category, to_messages(e)) for e in picked]
        if category in HOLDOUT_CATEGORIES:
            n_test = max(1, round(len(msgs) * HOLDOUT_RATIO))
            test += msgs[:n_test]
            train += msgs[n_test:]
            print(f"[{category}] 取 {len(msgs)} 条（train {len(msgs) - n_test}"
                  f" / test {n_test}）")
        else:
            train += msgs
            print(f"[{category}] 取 {len(msgs)} 条（train {len(msgs)}）")

    rng.shuffle(train)
    rng.shuffle(test)
    # 小语料（冒烟）时保证 train 非空；全量 1000 条时仍是 5%
    n_valid = min(max(20, round(len(train) * 0.05)), max(0, len(train) - 1))
    valid, train = train[:n_valid], train[n_valid:]

    expand = (lambda rows: rows) if args.no_expand else expand_rows
    train_rows, valid_rows = expand(train), expand(valid)
    # 留出集不展开：评估以整段会话为单位（也避免同一会话的前缀混进评估）
    test_rows = [{"category": c, "messages": m} for c, m in test]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test_holdout.jsonl", test_rows)
    print(f"\n会话数 train {len(train)} / valid {len(valid)} / holdout {len(test)}"
          f" → 展开样本 train {len(train_rows)} / valid {len(valid_rows)}"
          f" / holdout {len(test_rows)}（holdout 不展开）")
    print(f"输出 → {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
