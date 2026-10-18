"""M12.3：语料契约 → 训练格式（{"category","messages"}）。

**训练/推理同源（PRD 决策 C）**：context 渲染直接调服务端注入函数
`chat.context.render_rag_segment`，不在这里重写拼接。训练样本里的材料段与
线上注入段一字不差，否则 SFT 学到的格式线上对不上。

输入：`gen_law_qa.py` 产出的契约格式
  {"category","system","context":[{"ref","title","text"}],"question"或"messages","answer"}
输出：`--out-dir` 下的 train.jsonl / valid.jsonl，每行 {"category","messages":[...]}

用法：
  cd server
  python finetune/scripts/law_dataset_build.py \
      --input finetune/data/law/generated.jsonl --out-dir finetune/data/law/final
  # 冒烟：少量样本 + 指定留出比例
  python finetune/scripts/law_dataset_build.py --input ../spikes/law_corpus_preview.jsonl \
      --out-dir finetune/data/law/smoke --valid 2
"""
import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SERVER))

from chat.context import render_rag_segment  # noqa: E402
from config import config  # noqa: E402

ROLE = "你是一位中国大陆的劳动法咨询助手。"


def render_material(context: list[dict]) -> str:
    """复用生产注入函数：命中片段已是「《法》第X条：原文」形态。"""
    if not context:
        return ""
    hits = [{"filename": "", "content": f"{c['title']}：{c['text']}"}
            for c in context]
    return render_rag_segment(hits, config.rag_max_inject_tokens, len) or ""


def to_messages(record: dict) -> list[dict]:
    system = record.get("system") or ROLE
    material = render_material(record.get("context") or [])
    messages = [{"role": "system", "content": system}]
    if material:
        messages.append({"role": "system", "content": material})
    if record.get("messages"):  # 多轮：历史 + 末轮问题
        messages.extend(record["messages"])
        messages.append({"role": "assistant", "content": record["answer"]})
    else:
        messages.append({"role": "user", "content": record["question"]})
        messages.append({"role": "assistant", "content": record["answer"]})
    return messages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--valid", type=int, default=0,
                    help="留出条数；0 = 按 --valid-ratio 抽取")
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.input).open(encoding="utf-8")
               if l.strip()]
    rows = [{"category": r.get("category", "law_grounded_qa"),
             "messages": to_messages(r)} for r in records]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_valid = args.valid or max(1, int(len(rows) * args.valid_ratio))
    n_valid = min(n_valid, max(1, len(rows) - 1))
    valid, train = rows[:n_valid], rows[n_valid:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"输入 {len(records)} 条 → train {len(train)} / valid {len(valid)} → {out_dir}")
    # 同源抽查：材料段必须与注入格式一致（`[n] 《法》第X条：`）
    sample = next((m for r in train for m in r["messages"]
                   if m["content"].startswith("以下是知识库")), None)
    if sample:
        print("材料段样例：" + sample["content"][:120].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
