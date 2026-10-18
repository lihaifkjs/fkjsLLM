"""M12.2 消融跑批：同一知识库、同一 60 题集，只换「模型 × RAG 开关」。

组（PRD §7.1）：
  A base + RAG 关 / B base + RAG 开 / C law-14b + 关 / D law-14b + 开 /
  E fkjs-14b + RAG 开

**三个 stage 分开跑**，因为两个 Python 环境各缺一半依赖：
  prepare  检索 + 注入渲染 → prompts_<组>.json     （服务环境 myllm：有 fastembed）
  generate 加载模型逐题生成 → <组>_<后端>.jsonl     （.venv 跑 hf；myllm 跑 gguf）
  judge    自动判分 + 人工判分表 → summary.json / review.md（纯文本，任一环境）

同源约束（PRD 决策 C）：prepare 直接调生产函数
`rag.retriever.search` + `chat.context.render_rag_segment`，不复制拼接逻辑。

用法:
  cd server
  python spikes/law_ablation.py --stage prepare --groups A B
  finetune/.venv/Scripts/python.exe spikes/law_ablation.py --stage generate \
      --groups A B --backend hf
  python spikes/law_ablation.py --stage generate --groups E --backend gguf \
      --gguf ../fkjs-14b-v2-q4_k_m.gguf
  python spikes/law_ablation.py --stage judge
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
SERVER = SPIKE_DIR.parent
REPO = SERVER.parent
sys.path.insert(0, str(SERVER))

QUESTIONS = SPIKE_DIR / "law_eval_questions.json"
OUT_DIR = SPIKE_DIR / "law_ablation"
BASE_HF_MODEL = SERVER / "finetune" / "models" / "Qwen3-14B"
FKJS_GGUF = REPO / "fkjs-14b-v2-q4_k_m.gguf"
DEFAULT_LOG = SERVER / "finetune" / "law_ablation.log"

# 所有组共用同一句角色 prompt，保证「只换模型」成立；
# 领域能力差异应体现在回答里，不靠 system 提示喂出来。
ROLE_SYSTEM = "你是一位中国大陆的劳动法咨询助手。"

_CITE_RE = re.compile(r"\[(\d+)\]")
_LABEL_RE = re.compile(r"《[^》]{1,40}》第[零〇一二三四五六七八九十百]+条")
_ARTICLE_REF_RE = re.compile(r"第[零〇一二三四五六七八九十百]+条")
# think 段里模型会把材料条目称作「第一条/第二条」，那不是对外引用，判分要排除
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SENT_SPLIT = re.compile(r"[。；!?\n]")  # 判断「否认/建议」语境时按句切
_REFUSAL_HINTS = ("不凭记忆", "未检索到", "没有相关", "无法从", "材料中未",
                  "材料未", "不能提供", "无法提供", "超出范围", "超出本次",
                  "建议咨询", "建议向", "转介")


def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def load_questions(limit: int = 0) -> list[dict]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    return data[:limit] if limit else data


def gold_label(chunk_content: str) -> str:
    """chunk 首行到「：」即引用标签（与注入格式同源）。"""
    return chunk_content.split("：", 1)[0].strip()


# ---------------------------------------------------------------- stage 1
def stage_prepare(groups: list[str], questions: list[dict], log) -> None:
    """检索 + 渲染注入段，落 prompts_<组>.json（不含模型）。"""
    from chat.context import render_rag_segment
    from config import config
    from rag.retriever import search

    count_tokens = len  # 字符近似：服务环境没有生成模型的 tokenizer，按字符算只会更宽松

    for group in groups:
        rag_on = group in ("B", "D", "E")
        prepared = []
        for q in questions:
            turns = q.get("turns") or [q["q"]]
            hits = search(turns[-1]).hits if rag_on else []
            segment = render_rag_segment(
                hits, config.rag_max_inject_tokens, count_tokens) if rag_on else None
            messages = [{"role": "system", "content": ROLE_SYSTEM}]
            if segment:
                messages.append({"role": "system", "content": segment})
            prepared.append({
                "id": q["id"], "cat": q["cat"], "turns": turns,
                "messages": messages,
                "hits": [{"label": gold_label(h["content"]), "score": h["score"]}
                         for h in hits],
            })
        out = OUT_DIR / f"prompts_{group}.json"
        out.write_text(json.dumps(prepared, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        log(f"[{group}] RAG {'开' if rag_on else '关'}，"
            f"平均命中 {sum(len(p['hits']) for p in prepared) / len(prepared):.1f} 条 → {out}")


# ---------------------------------------------------------------- stage 2
class HfBackend:
    """transformers + bnb 4-bit，贪心解码（temperature=0），整批一次生成。

    `lora`：给 C/D 组用——不合并权重，直接在 4bit 底模上挂 PEFT adapter。
    合并 fp16 14B 需要 ~28GB 内存（本机只剩 16GB），挂 adapter 与训练时同条件，
    且省掉 28GB 磁盘与 10 分钟合并。
    """

    def __init__(self, model_dir: Path, max_new_tokens: int, batch: int,
                 max_gpu_gib: float | None, lora: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.torch, self.batch = torch, batch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs = {"device_map": {"": 0}}
        if max_gpu_gib:  # 显存被别的程序占着时，把部分层放到 CPU（慢但能跑）
            kwargs = {"device_map": "auto",
                      "max_memory": {0: f"{max_gpu_gib:.1f}GiB", "cpu": "64GiB"}}
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), quantization_config=quant, dtype=torch.bfloat16,
            attn_implementation="sdpa", **kwargs)
        if lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora))
        self.model = model.eval()

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        texts = [self.tok.apply_chat_template(m, tokenize=False,
                                             add_generation_prompt=True)
                 for m in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        n_prompt = enc["input_ids"].shape[1]
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(seq, skip_special_tokens=True)
                for seq in out[:, n_prompt:]]


class GgufBackend:
    """llama-cpp（本机 CPU 版，单条串行）：给只有 GGUF 的模型用。"""

    def __init__(self, model_path: Path, max_new_tokens: int):
        from llama_cpp import Llama, llama_supports_gpu_offload

        self.max_new_tokens = max_new_tokens
        self.llm = Llama(model_path=str(model_path), n_ctx=8192,
                         n_gpu_layers=-1 if llama_supports_gpu_offload() else 0,
                         verbose=False)

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        out = []
        for messages in prompts:
            resp = self.llm.create_chat_completion(
                messages=messages, temperature=0.0,
                max_tokens=self.max_new_tokens, stream=False)
            out.append(resp["choices"][0]["message"]["content"] or "")
        return out


def stage_generate(groups: list[str], backend_name: str, log, args) -> None:
    # 后端只建一次：同一份权重复用给多组。A/B 共用 base 模型时，若每组各建一次
    # 会在释放 4bit 权重后再加载时崩（Windows bnb，0xC0000005 实测）
    backend = None
    for group in groups:
        src = OUT_DIR / f"prompts_{group}.json"
        if not src.exists():
            log(f"[{group}] 缺少 {src.name}，先跑 --stage prepare")
            continue
        prepared = json.loads(src.read_text(encoding="utf-8"))
        if backend is None:
            if backend_name == "hf":
                backend = HfBackend(Path(args.hf_model), args.max_new_tokens,
                                    args.batch, args.max_gpu_gib or None,
                                    Path(args.lora) if args.lora else None)
            else:
                backend = GgufBackend(Path(args.gguf), args.max_new_tokens)
        log(f"[{group}] {backend_name} 生成 {len(prepared)} 题"
            f"（{Path(args.hf_model).name if backend_name == 'hf' else Path(args.gguf).name}）")

        t0 = time.perf_counter()
        rows = []
        single = [it for it in prepared if len(it["turns"]) == 1]
        multi = [it for it in prepared if len(it["turns"]) > 1]
        # 单轮题可整批解码（左 padding），吞吐约 2~3 倍；多轮只能逐轮串行
        batch = args.batch if backend_name == "hf" else 1
        for i in range(0, len(single), batch):
            chunk = single[i:i + batch]
            answers = backend.generate(
                [it["messages"] + [{"role": "user", "content": it["turns"][0]}]
                 for it in chunk])
            for it, answer in zip(chunk, answers):
                rows.append({**it, "answer": answer})
                log(f"  [{group}] {it['id']} {len(answer)} 字")
        for item in multi:
            messages = list(item["messages"])
            answer = ""
            for turn in item["turns"]:  # 多轮逐轮生成，前轮回答当历史（无 gold 参照）
                messages = messages + [{"role": "user", "content": turn}]
                answer = backend.generate([messages])[0]
                messages = messages + [{"role": "assistant", "content": answer}]
            rows.append({**item, "answer": answer})
            log(f"  [{group}] {item['id']} {len(answer)} 字（多轮）")
        out = OUT_DIR / f"{group}_{backend_name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"[{group}] 耗时 {hms(time.perf_counter() - t0)} → {out.name}")


# ---------------------------------------------------------------- stage 3
def parse_cites(answer: str) -> list[tuple[int, str | None]]:
    """解析 `[n]` 及其行内出处标签 → [(编号, 抄的标签 或 None)]。

    契约写法是标签在前（`《劳动合同法》第十九条[1]`），也容忍标签在后
    （`[1]《劳动合同法》第十九条`），两者都算「有出处」；都没有则记 None。
    """
    out = []
    for m in _CITE_RE.finditer(answer):
        before = answer[max(0, m.start() - 40):m.start()].rstrip()
        after = answer[m.end():m.end() + 40].lstrip()
        head = list(_LABEL_RE.finditer(before))
        label = None
        if head and before.endswith(head[-1].group(0)):
            label = head[-1].group(0)
        else:
            tail = _LABEL_RE.match(after)
            if tail:
                label = tail.group(0)
        out.append((int(m.group(1)), label))
    return out


def _article_texts() -> dict[str, str]:
    """标签 → 条文正文（用于识别材料内的交叉引用）。"""
    path = SERVER / "finetune" / "data" / "law" / "articles.json"
    if not path.exists():
        return {}
    return {a["label"]: a["text"] for a in json.loads(path.read_text(encoding="utf-8"))}


_ARTICLES = _article_texts()


def judge(answer: str, hits: list[dict], gold: list[str],
          expect_refuse: str | None) -> dict:
    """自动判分：引用合法性 / 出处标签一致性 / 材料外幻觉 / 拒答关键词。

    口径细节（都是实测后才收紧的）：
    - 只在正文里判，think 段把材料条目称作「第一条/第二条」不算幻觉
    - 材料正文里出现过的条号算材料内（第八十七条原文就写着「按第四十七条标准」）
    - 出处标签只要求**每个编号首次出现**时带；同一编号后续裸写 `[1]` 是正常写法
    """
    labels = [h["label"] for h in hits]
    body = _THINK_RE.sub("", answer)
    parsed = parse_cites(body)
    cites = [n for n, _ in parsed]
    cited_labels = [labels[n - 1] for n in cites if 1 <= n <= len(labels)]
    seen, label_mismatch, without_label = set(), [], []
    for n, lab in parsed:
        if n in seen:
            continue  # 后续裸引用不追究
        seen.add(n)
        if not lab:
            without_label.append(n)
        elif 1 <= n <= len(labels) and lab != labels[n - 1]:
            label_mismatch.append({"n": n, "wrote": lab, "material": labels[n - 1]})
    mentioned = {ref for h in hits for ref in _ARTICLE_REF_RE.findall(h.get("text", ""))}
    # 材料里没存正文时，用法条库补齐：第八十七条正文里就写着「按第四十七条标准」
    mentioned |= {ref for h in hits
                  for ref in _ARTICLE_REF_RE.findall(_ARTICLES.get(h["label"], ""))}
    # 只有「当成依据引用」才算材料外幻觉；下面这些句式是明确否认或建议补充材料，
    # 属于正确行为（实测 F16「建议补充第十七条」、C01「你提到的第八十四条不在材料中」
    # 曾被误判，8 题里 6 题都是这种）
    _DENY_HINTS = ("不在当前材料", "不在材料", "未收录", "没有该条", "无法引用",
                   "建议补充", "建议：补充", "建议:", "建议：", "请补充", "补充《",
                   "材料未", "核对", "无法确认")
    outside = []
    if labels:  # A/C 组本无材料，出现任何条号都不叫「材料外幻觉」，不统计
        for ref in _ARTICLE_REF_RE.findall(body):
            if ref in mentioned or any(ref in label for label in labels):
                continue
            # 取包含该条号的那一句，判断是否是否认/建议语气
            sentence = next((s for s in _SENT_SPLIT.split(body) if ref in s), body)
            if any(k in sentence for k in _DENY_HINTS):
                continue
            outside.append(ref)
    refused = any(h in answer for h in _REFUSAL_HINTS)
    res = {
        "cites": cites,
        "invalid_cites": [n for n in cites if not 1 <= n <= len(labels)],
        "cited_labels": cited_labels,
        "label_mismatch": label_mismatch,
        "cite_without_label": without_label,
        "out_of_material_refs": outside,
        "refused": refused,
    }
    if gold:  # 引对了条：编号全合法 + 至少引到一个 gold（多引相关条不算错，另记）
        res["cite_hit"] = (bool(cited_labels) and not res["invalid_cites"]
                           and any(l in gold for l in cited_labels))
        res["cite_extra"] = [l for l in cited_labels if l not in gold]
    else:
        res["cite_hit"] = None
    if expect_refuse:
        res["refuse_ok"] = refused and not cites
    return res


def stage_judge(log) -> None:
    questions = {q["id"]: q for q in load_questions()}
    rows = []
    for path in sorted(OUT_DIR.glob("*_*.jsonl")):
        group = path.name.split("_")[0]
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            q = questions[r["id"]]
            rows.append({"group": group, **r,
                         **judge(r["answer"], r["hits"], q.get("gold") or [],
                                 q.get("expect_refuse"))})
    if not rows:
        log("没有可判分的结果文件")
        return

    def pct(hit: int, total: int) -> str:
        return f"{hit}/{total} = {hit / total * 100:.0f}%" if total else "-"

    summary = {}
    for group in sorted({r["group"] for r in rows}):
        g = [r for r in rows if r["group"] == group]
        cite_rows = [r for r in g if r["cat"] in ("fact", "cite", "multiturn")]
        refuse_rows = [r for r in g if questions[r["id"]].get("expect_refuse")]
        summary[group] = {
            "n": len(g),
            "cite_hit": pct(sum(1 for r in cite_rows if r["cite_hit"]), len(cite_rows)),
            "invalid_cite_rows": sum(1 for r in g if r["invalid_cites"]),
            "cite_extra_rows": sum(1 for r in g if r.get("cite_extra")),
            "label_mismatch_rows": sum(1 for r in g if r["label_mismatch"]),
            "cite_without_label_rows": sum(1 for r in g if r["cite_without_label"]),
            "halluc_rows": sum(1 for r in g if r["out_of_material_refs"]),
            "halluc_refs_total": sum(len(r["out_of_material_refs"]) for r in g),
            "refuse_ok": pct(sum(1 for r in refuse_rows if r.get("refuse_ok")),
                             len(refuse_rows)),
        }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐题判分结果落盘（jsonl 里只有答案，判分在这之后算，需要单独存一份）
    with (OUT_DIR / "judged.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# M12.2 消融人工判分表", "",
             "> 自动列由脚本算；「结论」「拒答」需人工填 √ / ×。", ""]
    for r in rows:
        lines += [f"## [{r['group']}] {r['id']}（{r['cat']}）", "",
                  f"- 命中材料：{', '.join(h['label'] for h in r['hits']) or '（无）'}",
                  f"- 引用：{r['cites'] or '无'}；材料外条号：{r['out_of_material_refs'] or '无'}",
                  f"- 引用出处：抄错 {len(r['label_mismatch'])} 处、无出处 {len(r['cite_without_label'])} 处",
                  "- 结论：☐ 正确 ☐ 错误　拒答：☐ 是 ☐ 否", "",
                  "```", r["answer"].strip(), "```", ""]
    (OUT_DIR / "review.md").write_text("\n".join(lines), encoding="utf-8")
    log("自动指标 " + json.dumps(summary, ensure_ascii=False))
    log(f"人工判分表 {OUT_DIR / 'review.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("prepare", "generate", "judge"))
    ap.add_argument("--groups", nargs="+", default=["A", "B"],
                    choices=("A", "B", "C", "D", "E"))
    ap.add_argument("--backend", choices=("hf", "gguf"), default="hf")
    ap.add_argument("--hf-model", default=str(BASE_HF_MODEL))
    ap.add_argument("--lora", default="",
                    help="PEFT adapter 目录（C/D 组用，不合并权重直接挂载）")
    ap.add_argument("--gguf", default=str(FKJS_GGUF))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-gpu-gib", type=float, default=0,
                    help="限制 GPU 显存（GiB），0=全部；被别的程序占显存时用它")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8")

    def log(msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    if args.stage == "prepare":
        stage_prepare(args.groups, load_questions(args.limit), log)
    elif args.stage == "generate":
        stage_generate(args.groups, args.backend, log, args)
    else:
        stage_judge(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
