"""M12.4：把 LoRA adapter 合并进底模（**逐 shard**，不吃满内存）。

为什么不直接 `merge_and_unload()`：那要把 14B 的 bf16 权重（约 28GB）整个读进内存，
本机可用内存只有 16GB 左右，必 OOM。这里改成按 shard 处理，峰值内存 ≈ 单个 shard
（3.7GB）+ 一个张量的 fp32 副本。

数学：LoRA 的增量是 ΔW = (B @ A) * (alpha / r)，加到原权重上
  W' = W + ΔW
其中 A:[r, in]、B:[out, r]，alpha/r 是本项目的缩放（32/16 = 2.0）。
adapter 的键名形如 `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`，
底模对应 `model.layers.0.mlp.down_proj.weight` —— 映射靠去前缀/去后缀完成。

用法：
  python scripts/merge_lora_sharded.py \
      --base models/Qwen3-14B --adapter outputs/m12-law-v1/final --out models/law-14b-merged
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COPY_SUFFIXES = (".json", ".txt", ".md", ".gitattributes")
ADAPTER_PREFIX = "base_model.model."
ADAPTER_A_SUFFIX = ".lora_A.weight"  # 前导点不能漏：键是 "...down_proj.lora_A.weight"


def load_adapter(adapter_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{底模参数名: (lora_A, lora_B)}。"""
    path = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        pairs = {}
        for key in keys:
            if not key.endswith("lora_A.weight"):
                continue
            root = key[len(ADAPTER_PREFIX):-len(ADAPTER_A_SUFFIX)]  # 例：model.layers.0.mlp.down_proj
            b_key = f"{ADAPTER_PREFIX}{root}.lora_B.weight"
            if b_key not in keys:
                raise RuntimeError(f"缺配对的 lora_B：{key}")
            pairs[f"{root}.weight"] = (f.get_tensor(key), f.get_tensor(b_key))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, adapter_dir, out = Path(args.base), Path(args.adapter), Path(args.out)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = load_adapter(adapter_dir)
    print(f"adapter 覆盖 {len(pairs)} 个参数，缩放 alpha/r = {scale}")

    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():  # 配置/分词器等小文件直接复制
        if f.suffix in COPY_SUFFIXES or f.name == ".gitattributes":
            shutil.copy2(f, out / f.name)

    new_map, applied, t0 = {}, 0, time.perf_counter()
    for shard in shards:
        tensors, merged_here = {}, 0
        with safe_open(str(base / shard), framework="pt") as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]
                    # fp32 里算增量再落回 bf16：避免 bf16 累加精度损失
                    delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
                    w = (w.to(torch.float32) + delta).to(torch.bfloat16)
                    merged_here += 1
                tensors[key] = w
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        for key in tensors:
            new_map[key] = shard
        applied += merged_here
        print(f"  {shard}: 合并 {merged_here} 个参数 | 累计 {applied}/{len(pairs)}"
              f" | {time.perf_counter() - t0:.0f}s", flush=True)

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": new_map}, indent=1),
        encoding="utf-8")
    print(f"完成：{out}（合并 {applied} 个参数，{time.perf_counter() - t0:.0f}s）")
    if applied != len(pairs):
        print(f"!! 警告：adapter 有 {len(pairs)} 个参数没全部用上")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
