"""批次 3（Windows 正式训练）：Qwen3-14B QLoRA 训练。

对应 M8 计划 §5：4-bit NF4 底模冻结 + LoRA，transformers + peft + trl。
与 Mac MLX 脚本（train_loop.sh）互不依赖，数据共用 data/final/*.jsonl。

用法（venv 内）:
  python train_windows.py --smoke                    # 冒烟：3 步验证管线
  python train_windows.py                            # 全量训练（2 epoch，~120 步）
  python train_windows.py --epochs 3 --lr 2e-4       # 全量调优

产出: outputs/<run_name>/checkpoint-*（LoRA adapter）+ runs/ 训练日志
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForSeq2Seq)
from trl import SFTConfig, SFTTrainer

REPO_FINETUNE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_FINETUNE / "models" / "Qwen3-14B"
DEFAULT_DATA = REPO_FINETUNE / "data" / "final"


def pretokenize(path: Path, tokenizer, max_seq: int) -> tuple[Dataset, int]:
    """{"messages": [...]} JSONL → 预分词 Dataset（input_ids + labels）。

    Qwen3 官方模板无 {% generation %} 标记，trl 的 assistant_only_loss 不可用，
    改为手工掩码：prompt（除最后一条 assistant 外的消息 + generation prompt）
    单独分词，作为前缀从整段 token 中切出，前缀之外计 loss。
    整段与前缀拼接不一致的样本直接丢弃（BPE 跨界合并，数量极少）。
    """
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    input_ids_list, labels_list, dropped = [], [], 0
    for r in rows:
        msgs = r["messages"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) > max_seq:
            dropped += 1
            continue
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    ds = Dataset.from_list([{"input_ids": i, "labels": l}
                            for i, l in zip(input_ids_list, labels_list)])
    return ds, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--output-root", default=str(REPO_FINETUNE / "outputs"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=640)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--resume", default=None, help="checkpoint 目录，断点续训")
    ap.add_argument("--smoke", action="store_true", help="只跑 3 步验证管线")
    args = ap.parse_args()

    run_name = args.run_name or f"qlora-{datetime.now():%m%d-%H%M}"
    if args.smoke:
        run_name += "-smoke"
    out_dir = Path(args.output_root) / run_name

    if not torch.cuda.is_available():
        print("!! CUDA 不可用，QLoRA 训练无法进行", file=sys.stderr)
        return 1
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_gb:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 4-bit NF4 冻结底模：14B ≈ 8GB 显存，给激活/优化器留 ~4GB
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # 训练期关闭 KV cache，配合梯度检查点

    # M8 计划 §5 基线：r=16 / alpha=32 / dropout 0.05，覆盖全部线性层
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    data_dir = Path(args.data)
    train_ds, drop_tr = pretokenize(data_dir / "train.jsonl", tokenizer, args.max_seq)
    valid_ds, drop_va = pretokenize(data_dir / "valid.jsonl", tokenizer, args.max_seq)
    print(f"train {len(train_ds)} 条（丢弃 {drop_tr}）| valid {len(valid_ds)} 条（丢弃 {drop_va}）")

    max_steps = 3 if args.smoke else args.max_steps
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=args.batch,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        # 数据已预分词（input_ids + labels），跳过 trl 的聊天模板处理
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=[],
        seed=42,
    )
    # 预分词数据的标准拼板 collator：右 pad input_ids，labels 补 -100
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=valid_ds, peft_config=peft_cfg,
                         data_collator=collator, processing_class=tokenizer)

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"训练完成 → {out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
