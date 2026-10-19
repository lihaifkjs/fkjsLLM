"""画 M12/M13 的 loss 曲线（v3 vs v4），并（可选）画 PPL 对比。

loss 数据直接解析训练日志里的 HF 日志行：
  {'loss': ..., 'learning_rate': ..., 'epoch': ...}
  {'eval_loss': ..., 'epoch': ...}
无需 GPU，用 myllm 的 matplotlib 即可。

用法：
  python server/spikes/plot_law_curves.py loss
  python server/spikes/plot_law_curves.py ppl --json server/spikes/law_ppl.json
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPIKE = Path(__file__).resolve().parent
REPO = SPIKE.parents[1]
ASSETS = REPO / "docs" / "assets"
LOGS = {
    "v3（纯法律 3ep）": SPIKE.parent / "finetune" / "data" / "law" / "train_v3.out",
    "v4（混合 2ep）": SPIKE.parent / "finetune" / "data" / "law" / "train_v4_full.out",
}
_NUM = re.compile(r"'(\w+)':\s*([0-9.eE+-]+)")

# 中文字体：Windows 自带雅黑，缺失时退回默认（不至于崩）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_log(path: Path) -> str:
    """PowerShell 的 `*>` 重定向在 5.1 下写 UTF-16LE，python 的 `>` 写 UTF-8。
    这里按 BOM/空字节探测，两种都能读。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if b"\x00" in raw[:4000]:
        return raw.decode("utf-16-le", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_log(path: Path):
    """返回 (train[(step, loss)], eval[(step, loss)])；step 由日志出现顺序推算。"""
    train, evals, step = [], [], 0
    for line in _read_log(path).splitlines():
        line = line.strip()
        if "'eval_loss'" in line:
            m = re.search(r"'eval_loss':\s*([0-9.eE+-]+)", line)
            if m:
                evals.append((step, float(m.group(1))))
        elif "'loss'" in line:
            step += 1
            m = re.search(r"'loss':\s*([0-9.eE+-]+)", line)
            if m:
                train.append((step, float(m.group(1))))
    return train, evals


def smooth(values, k=7):
    out, buf = [], []
    for v in values:
        buf.append(v)
        if len(buf) > k:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def plot_loss(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"v3（纯法律 3ep）": "#7f8c8d", "v4（混合 2ep）": "#2980b9"}
    for label, path in LOGS.items():
        if not path.exists():
            continue
        train, evals = parse_log(path)
        if not train:
            continue
        xs = [s for s, _ in train]
        ys = [v for _, v in train]
        c = colors.get(label, "C0")
        axes[0].plot(xs, ys, color=c, alpha=0.25, linewidth=0.8)
        axes[0].plot(xs, smooth(ys), color=c, linewidth=2, label=f"{label}  平滑(7)")
        if evals:
            axes[1].plot([s for s, _ in evals], [v for _, v in evals],
                         color=c, marker="o", linewidth=2, label=label)

    axes[0].set_title("训练 loss（细线=原始，粗线=7 步滑动平均）")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("验证 loss（法律验证集 20 条）")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("eval_loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("M12/M13 法律模型训练曲线 · v3（纯法律）vs v4（掺 replay + 身份样本）",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"→ {out}")


def plot_ppl(data_path: Path, out: Path):
    data = json.loads(data_path.read_text(encoding="utf-8"))  # {"labels":[...], "series":{name:[...]}}
    labels, series = data["labels"], data["series"]
    fig, ax = plt.subplots(figsize=(var_width := max(7, len(labels) * 1.1), 4.6))
    import numpy as np
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("perplexity（越低越好）")
    ax.set_title("通用文本困惑度 · 底模 / v3 / v4")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"→ {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("loss", "ppl"))
    ap.add_argument("--json", default=str(SPIKE / "law_ppl.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.what == "loss":
        plot_loss(Path(args.out) if args.out else ASSETS / "m13-v4-loss.png")
    else:
        plot_ppl(Path(args.json), Path(args.out) if args.out else ASSETS / "m13-v4-ppl.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
