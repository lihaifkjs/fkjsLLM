"""M12.4：法律语料 + M11 replay 混合数据集（修「通用/身份能力被挤压」）。

背景：v3 只用纯法律语料训练（139 条），实测问「你是谁」会陷入「材料里有没有」的自我纠结，
还把「不在材料范围就拒」的规则泛化到自我介绍上。PRD §5.3 原本就要求掺 replay
（通用 300 + tool 100，沿用 M11 数据），这一步被漏了。

混合比例：法律 : replay ≈ 1 : 2（法律 119 条，replay 约 320 条）。法律任务窄，
replay 占多不会稀释格式学习，但能挡住能力漂移。

用法（在 server 目录）：
  python finetune/scripts/law_dataset_mix.py
  → finetune/data/law/final_mix/{train,valid}.jsonl
"""
import argparse
import json
import random
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2]
LAW = SERVER / "finetune" / "data" / "law" / "final"
CURATED = SERVER / "finetune" / "data" / "law" / "curated_identity.jsonl"
M11 = SERVER / "finetune" / "data" / "final"
OUT = SERVER / "finetune" / "data" / "law" / "final_mix"

# M11 各类抽多少（identity/bypass/boundary/multiturn 全要，replay 抽样控总量）
SAMPLE = {
    "identity": 66,
    "bypass": 45,
    "boundary": 28,
    "multiturn": 10,
    "replay_general": 120,
    "replay_tool": 50,
}
SEED = 42


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    # 法律样本重复次数：>1 相当于按条数加权，把法律 token 占比从 46% 拉到 60%+
    # （v2 的负结果说明法律侧样本偏少会掉拒答率，宁可多给）
    ap.add_argument("--law-replicas", type=int, default=1)
    ap.add_argument("--out", default="final_mix")
    args = ap.parse_args()

    rng = random.Random(SEED)
    law_train, law_valid = load(LAW / "train.jsonl"), load(LAW / "valid.jsonl")
    m11_train = load(M11 / "train.jsonl")

    by_cat: dict[str, list[dict]] = {}
    for r in m11_train:
        by_cat.setdefault(r.get("category", "?"), []).append(r)

    replay, stat = [], {}
    for cat, n in SAMPLE.items():
        pool = by_cat.get(cat, [])
        rng.shuffle(pool)
        picked = pool[:n]
        replay.extend(picked)
        stat[cat] = len(picked)

    law_expanded = law_train * max(1, args.law_replicas)
    train = law_expanded + replay
    # 带法律 system 的身份/边界样本：M11 的 identity 都是「无 system」，
    # 而线上问「你是谁」时必然带 system（角色 + 可能有空材料提示）——
    # 缺这层对齐，模型会把「材料不足就拒」的规则套到自我介绍上（实测见 docs）
    curated = [json.loads(l) for l in CURATED.open(encoding="utf-8") if l.strip()]
    train += curated * max(1, args.law_replicas)
    rng.shuffle(train)

    out = SERVER / "finetune" / "data" / "law" / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", law_valid)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"法律 {len(law_train)}×{args.law_replicas} + replay {len(replay)} "
          f"+ 身份 {len(curated)}×{args.law_replicas} = {len(train)}；valid {len(law_valid)}")
    print("replay 明细:", stat)
    print(f"→ {out}")
    # 抽查身份样本确实在里面
    hit = [r for r in train if r.get("category") == "identity"][:1]
    if hit:
        print("样例 identity:", hit[0]["messages"][-1]["content"][:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
