"""M12.5 demo（GPU 版）：复用训练/评测那套 transformers + 4bit 跑四幕。

为什么不走服务端：服务端绑 llama-cpp-python，而它的 CUDA wheel 停在 0.3.4（不认 Qwen3），
本机又没有 MSVC/nvcc 编译新版本。训练/评测用的 transformers 栈本来就支持 Qwen3 + CUDA，
所以 demo 直接复用它——**不修服务端也能演出四幕**，代价是不能在浏览器里点。

两段式（两个 Python 环境各缺一半依赖，与 law_ablation.py 同构）：
  --stage prepare   （服务环境 myllm，有 fastembed）：检索 + 渲染注入段 → demo_prompts.json
  --stage generate  （训练环境 .venv，有 transformers）：加载 4bit 底模（±LoRA）生成 → demo_out.json

用法：
  cd server
  python spikes/law_demo_gpu.py --stage prepare
  finetune/.venv/Scripts/python.exe spikes/law_demo_gpu.py --stage generate
"""
import argparse
import json
import os
import time
from pathlib import Path

import sys
SPIKE_DIR = Path(__file__).resolve().parent
SERVER = SPIKE_DIR.parent
REPO = SERVER.parent
sys.path.insert(0, str(SERVER))

# 当前 myllm 里装的是 CUDA 版 llama-cpp-python（0.3.4），导入它需要 torch 自带的
# CUDA 运行库在 PATH 里；prepare 阶段会间接 import model_manager，所以先补 PATH。
_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.is_dir():
    os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ.get("PATH", "")

BASE_HF = SERVER / "finetune" / "models" / "Qwen3-14B"
LORA = SERVER / "finetune" / "outputs" / "m12-law-v3" / "final"
PROMPTS = SPIKE_DIR / "demo_prompts.json"
OUT = SPIKE_DIR / "demo_out_gpu.json"
ROLE = "你是一位中国大陆的劳动法咨询助手。"

QUESTION = "我签的是三年期合同，公司说试用期八个月，这样合法吗？"
VIRTUAL_DOC = "demo_virtual.txt"
VIRTUAL_TEXT = (
    "《劳动合同法》第一百三十二条：用人单位以电子形式订立劳动合同的，"
    "应当保证劳动者可以随时下载、打印完整合同文本；未提供的，劳动者有权要求补发。\n"
)
VIRTUAL_Q = "公司用电子形式订立劳动合同，但不给我可以下载打印的完整合同文本，可以吗？"

ACTS = [
    {"id": "1", "title": "① 原版模型 · 无材料（裸考）", "rag": False, "lora": False,
     "question": QUESTION},
    {"id": "2", "title": "② 原版模型 · 有材料", "rag": True, "lora": False,
     "question": QUESTION},
    {"id": "3", "title": "③ law-14b · 有材料", "rag": True, "lora": True,
     "question": QUESTION},
    {"id": "4", "title": "④ law-14b · 材料里是虚构条文", "rag": True, "lora": True,
     "question": VIRTUAL_Q, "virtual": True},
]


def stage_prepare() -> int:
    """服务环境跑：检索 + 渲染注入段（含第 ④ 幕临时注入的虚构条文）。"""
    from rag.chunker import is_article_chunk  # noqa: F401  仅确认环境可用
    from rag.ingest import ingest_document
    from rag.retriever import search
    from chat.context import RAG_NO_HIT_HINT, render_rag_segment
    from config import config
    from session_store.sqlite_store import session_store

    for d in session_store.list_documents():
        if d.filename == VIRTUAL_DOC:
            session_store.delete_document(d.id)

    doc_id = None
    prepared = []
    for act in ACTS:
        if act.get("virtual"):
            doc_id = ingest_document(VIRTUAL_DOC, VIRTUAL_TEXT.encode("utf-8"),
                                     chunk_mode="article").id
            print(f"  注入虚构条文 #{doc_id}")
        hits = search(act["question"]).hits if act["rag"] else []
        segment = None
        if act["rag"]:
            segment = render_rag_segment(hits, config.rag_max_inject_tokens, len) \
                if hits else RAG_NO_HIT_HINT
        messages = [{"role": "system", "content": ROLE}]
        if segment:
            messages.append({"role": "system", "content": segment})
        messages.append({"role": "user", "content": act["question"]})
        prepared.append({**act, "messages": messages,
                         "materials": [h["content"].split("：")[0] for h in hits]})
        print(f"  [幕{act['id']}] 材料 {len(hits)} 条: {[h['content'].split('：')[0] for h in hits]}")

    PROMPTS.write_text(json.dumps(prepared, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    if doc_id:
        session_store.delete_document(doc_id)
        print("  已移除虚构条文，知识库恢复原状")
    print(f"→ {PROMPTS}")
    return 0


class TransformersBackend:
    """4bit 底模 + 可选 LoRA adapter（与训练/评测同源）。"""

    def __init__(self, lora: Path | None, max_new_tokens: int = 640):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(str(BASE_HF))
        model = AutoModelForCausalLM.from_pretrained(
            str(BASE_HF),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        if lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora))
        self.model = model.eval()

    def generate(self, messages: list[dict]) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def stage_generate() -> int:
    prepared = json.loads(PROMPTS.read_text(encoding="utf-8"))
    results = []
    for lora_flag in (False, True):  # 两幕一组：先不挂 adapter，再挂 adapter
        group = [a for a in prepared if a["lora"] == lora_flag]
        if not group:
            continue
        backend = TransformersBackend(LORA if lora_flag else None)
        for act in group:
            t0 = time.time()
            answer = backend.generate(act["messages"])
            body = answer.split("</think>")[-1].strip() if "</think>" in answer else answer.strip()
            results.append({"act": act["title"], "materials": act["materials"],
                            "answer": body, "seconds": round(time.time() - t0, 1)})
            print(f"  [幕{act['id']}] {len(body)} 字 / {results[-1]['seconds']}s", flush=True)
        del backend
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("prepare", "generate"))
    args = ap.parse_args()
    return stage_prepare() if args.stage == "prepare" else stage_generate()


if __name__ == "__main__":
    raise SystemExit(main())
