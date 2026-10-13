#!/bin/bash
# M8 Mac 冒烟训练：崩溃自动续训包装器。
# Metal Internal Error（0000000e）在本机随机发生（GPU 工作集压力），
# checkpoint 每 100 步落盘，崩溃后从最新 adapters.safetensors 续训，直到 mlx_lm 正常退出（0）。
# 用法: bash server/finetune/scripts/train_loop.sh [iters] [lr]

set -u
cd "$(dirname "$0")/../../.."   # 仓库根目录

ITERS="${1:-1000}"
LR="${2:-1e-4}"
LOG="server/finetune/train_lr1e4.log"
ADAPTER_DIR="server/finetune/adapters"

for attempt in $(seq 1 20); do
    echo "===== attempt $attempt $(date '+%H:%M:%S') =====" >> "$LOG"
    server/finetune/.venv/bin/python -m mlx_lm lora \
        --model server/finetune/models/Qwen3-14B-4bit \
        --train \
        --data server/finetune/data/final \
        --fine-tune-type lora \
        --num-layers 16 \
        --batch-size 2 \
        --iters "$ITERS" \
        --learning-rate "$LR" \
        --grad-checkpoint \
        --save-every 25 \
        --resume-adapter-file "$ADAPTER_DIR/adapters.safetensors" \
        --adapter-path "$ADAPTER_DIR" \
        --seed 42 >> "$LOG" 2>&1 && { echo "===== 训练完成 $(date '+%H:%M:%S') =====" >> "$LOG"; exit 0; }
    echo "===== 崩溃（exit=$?），15s 后从最新 checkpoint 续训 =====" >> "$LOG"
    sleep 15
done
echo "===== 重试 20 次仍失败，放弃 =====" >> "$LOG"
exit 1
