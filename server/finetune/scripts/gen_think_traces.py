"""M11：用 base 模型给通用语料生成 think 轨迹。

把 single-turn 语料的 answer 从「直答」改写为
`<think>\\n思考\\n</think>\\n\\n正文`，让 SFT 重新学会思考（背景见 docs/M11 SFT再训练.md）。
**身份三类（identity/bypass/boundary）与 replay_tool 不要用本脚本**——前者的身份口径
是直答，后者要保持 /no_think 的形态可判别性（M9 决策轮依赖）。

四个实现要点：
1. 思考与正文都用 base 模型自己生成的（自洽）；不用它的思考配别人的答案
2. 已含 `</think>` 的条目默认跳过 → 长跑可中断续跑，重跑不会重复生成
3. 未闭合 / 思考为空 / 超过 --max-think-chars 的条目**不采用**（保留原直答并计入
   统计）——宁可少几条 thinking，也不把半截推理或超长样本塞进 loss
4. **每批进度即时落盘 + 落 log**：output 每批整体重写（未改动则跳过），log 每批
   flush，进程被杀也只丢最后一批，重跑接着干

两个后端（--backend）：
- `llamacpp`（默认）：CPU/GGUF 走原路径，单条串行。本机实测 **71s/条**（14B Q4_K_M，
  `gpu_offload=False` 的 CPU wheel），800 条约 16 小时
- `hf`：transformers + bitsandbytes 4-bit，**一次解码一批**（左 padding）。本机 4070Ti
  实测 batch=8 **16.3s/条**（0.051 s/tok），800 条约 3.6 小时；显存 4-bit 底模 9.3GB
  + batch8 KV ≈ 11GB / 12GB

`--system-hint`（默认开启）只在**生成时**给模型一个 system 消息要求思考简洁；
它不会写进样本（样本仍只有 user + assistant），所以不引入 train/serve 偏斜。
实测：不加提示时 base 的思考中位 695 字，会让样本超出 train_windows 的 max_seq。

用法:
  # GPU 后端全量（原地改写；data/ 已入 git，改坏了 git checkout 还原）
  python gen_think_traces.py --backend hf --input data/curated/replay_general.jsonl \
      --output data/curated/replay_general.jsonl
  # 冒烟（10 条，写到临时目录）
  python gen_think_traces.py --backend hf --limit 10 \
      --input data/curated/replay_general.jsonl --output data/smoke/replay_general.jsonl
  # CPU/GGUF 后端（保持原行为）
  python gen_think_traces.py --input data/curated/replay_general.jsonl \
      --output data/curated/replay_general.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # 仓库根目录（GGUF 所在处）
FINETUNE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "Qwen3-14B-Q4_K_M.gguf"  # llama.cpp 后端的 base 原版，默认就带 think
DEFAULT_HF_MODEL = FINETUNE / "models" / "Qwen3-14B"  # HF 后端底模（M8 训练用的同一份）
DEFAULT_LOG = FINETUNE / "gen_think_traces.log"  # 已被 .gitignore 的 server/finetune/*.log 覆盖
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


class LlamacppGenerator:
    """CPU/GGUF 路径：单条串行（llama.cpp 的 chat 接口没有批量）。"""

    def __init__(self, model_path: str, max_tokens: int):
        from llama_cpp import Llama, llama_supports_gpu_offload

        self.max_tokens = max_tokens
        self.llm = Llama(model_path=model_path, n_ctx=8192,
                         n_gpu_layers=-1 if llama_supports_gpu_offload() else 0,
                         verbose=False)

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        out = []
        for messages in prompts:
            resp = self.llm.create_chat_completion(
                messages=messages, temperature=0.6, top_p=0.95, top_k=20,
                max_tokens=self.max_tokens, stream=False)
            out.append(resp["choices"][0]["message"]["content"] or "")
        return out


class HfGenerator:
    """GPU 路径：transformers + bitsandbytes 4-bit，整批一次解码（左 padding）。"""

    def __init__(self, model_path: str, max_tokens: int, load_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # 关掉分片加载进度条，免得 log 里混 stderr 噪声
        hf_logging.set_verbosity_error()
        self.torch = torch
        self.max_tokens = max_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"  # 左 padding：生成段在所有样本里都从同一位置开始
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ) if load_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0})
        self.model.eval()

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        n_prompt = enc["input_ids"].shape[1]  # 左 padding 后所有样本的公共前缀长度
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_tokens, do_sample=True,
                temperature=0.6, top_p=0.95, top_k=20,
                pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(seq, skip_special_tokens=True)
                for seq in out[:, n_prompt:]]


class ProgressLog:
    """进度同时写终端与 log 文件（每行 flush，进程被杀也留痕）。"""

    def __init__(self, path: Path | None):
        self.file = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("a", encoding="utf-8")

    def __call__(self, message: str, stamp: bool = True):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}" if stamp else message
        print(line, flush=True)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()


def write_jsonl(path: Path, entries: list[dict]):
    """整体重写（先写临时文件再替换，避免半截文件）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(path)


def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="可与 --input 相同（原地改写）")
    ap.add_argument("--backend", choices=("llamacpp", "hf"), default="llamacpp",
                    help="llamacpp=CPU/GGUF 单条；hf=GPU 4-bit 批量")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="llamacpp 后端的 gguf")
    ap.add_argument("--hf-model", default=str(DEFAULT_HF_MODEL), help="hf 后端的底模目录")
    ap.add_argument("--hf-4bit", action=argparse.BooleanOptionalAction, default=True,
                    help="hf 后端是否 4-bit 量化加载（默认开）")
    ap.add_argument("--batch", type=int, default=0,
                    help="0=按后端取默认（llamacpp 1 / hf 8）；显存紧张可调小")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-think-chars", type=int, default=500,
                    help="思考段超长则弃用该条（约束训练样本总长）")
    ap.add_argument("--system-hint", default=DEFAULT_HINT,
                    help="生成期专用 system 提示（压短思考）；传空串关闭")
    ap.add_argument("--save-every", type=int, default=1, help="每几个批次落盘一次 output")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="进度日志（追加）；空串关闭文件日志")
    ap.add_argument("--dry-run", action="store_true", help="只统计待处理条数，不加载模型")
    args = ap.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    entries = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    todo = [i for i, e in enumerate(entries) if "</think>" not in e["answer"]]
    if args.limit:
        todo = todo[:args.limit]
    batch = args.batch or (8 if args.backend == "hf" else 1)
    print(f"{in_path.name}: 共 {len(entries)} 条，待生成 {len(todo)} 条"
          f"（已含 think 跳过 {len(entries) - len(todo)} 条）")
    if args.dry_run or not todo:
        return 0

    log = ProgressLog(Path(args.log) if args.log else None)
    log("=" * 72, stamp=False)
    log(f"开始：backend={args.backend} batch={batch} 待生成 {len(todo)} 条"
        f" | in={in_path} out={out_path}")
    log(f"参数：max_tokens={args.max_tokens} max_think_chars={args.max_think_chars} "
        f"system_hint={'on' if args.system_hint else 'off'} "
        f"model={args.hf_model if args.backend == 'hf' else args.model}")

    t_load = time.perf_counter()
    if args.backend == "hf":
        gen = HfGenerator(args.hf_model, args.max_tokens, load_4bit=args.hf_4bit)
    else:
        gen = LlamacppGenerator(args.model, args.max_tokens)
    log(f"模型加载完成，用时 {hms(time.perf_counter() - t_load)}")

    stats, rejected = [], []  # 成功/未采用的思考长度，用于定 --max-think-chars
    failed = {"未思考（直答）": 0, "think 未闭合（疑似超长截断）": 0,
              "思考为空": 0, "思考过长": 0}
    t0 = time.perf_counter()
    done, next_log, dirty = 0, 0, False
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        prompts = []
        for idx in chunk:
            messages = ([{"role": "system", "content": args.system_hint}]
                        if args.system_hint else [])
            messages.append({"role": "user", "content": entries[idx]["question"]})
            prompts.append(messages)
        t_batch = time.perf_counter()
        outputs = gen.generate(prompts)
        for idx, text in zip(chunk, outputs):
            parsed = split_output(text)
            entry = entries[idx]
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
                    dirty = True
        done += len(chunk)
        if dirty and (start // batch + 1) % args.save_every == 0:
            write_jsonl(out_path, entries)
            dirty = False
        # 进度：批量后每批一条，单条模式每 10 条一条
        if done >= next_log or done == len(todo):
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (len(todo) - done)
            log(f"[{done}/{len(todo)}] 成功 {len(stats)} 失败 {sum(failed.values())}"
                f" | 本批 {time.perf_counter() - t_batch:.0f}s / 均 {elapsed / done:.1f}s 每条"
                f" | 已用 {hms(elapsed)} | ETA {hms(eta)}")
            next_log = done + (batch if batch > 1 else 10)
    if dirty:
        write_jsonl(out_path, entries)

    log(f"生成 {len(stats)} 条 → {out_path}，总用时 {hms(time.perf_counter() - t0)}")
    if stats:
        stats.sort()
        log(f"思考长度 中位 {statistics.median(stats):.0f} / p95 "
            f"{stats[int(len(stats) * 0.95)]} / max {stats[-1]} 字")
    if rejected:
        rejected.sort()
        log(f"过长未采用 {len(rejected)} 条：中位 "
            f"{statistics.median(rejected):.0f} / max {rejected[-1]} 字"
            f"（当前上限 {args.max_think_chars}，可 --max-think-chars 放宽，"
            f"但需同步确认 train_windows 的 --max-seq）")
    log(f"未采用：{failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
