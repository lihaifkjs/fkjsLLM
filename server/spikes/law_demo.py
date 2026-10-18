"""M12.5 demo：四幕脚本（可复现，走生产检索/注入链路）。

四幕对应 PRD §7.4：
  ① 原版模型裸考（无材料）—— 编条文号 / 凭记忆答
  ② 原版模型 + 材料 —— 有条文却不会引用，且混入记忆内容
  ③ law-14b + 材料 —— 结论 + 依据 + 带出处的引用
  ④ 往知识库塞一条**真实法条里没有**的规定 → 重问，答案跟着材料变
     （证明知识在库里、不在权重里）

第 ④ 幕是关键：它同时反驳「模型背下来了」这个质疑——因为那条规则在真实法律里不存在，
模型不可能背过，只能从材料读出来。

用法（本地 CPU 推理，一题约 1 分钟）:
  cd server && python spikes/law_demo.py
  python spikes/law_demo.py --acts 1 2 3      # 只跑指定幕
"""
import argparse
import json
import time
from pathlib import Path

import sys
SERVER = Path(__file__).resolve().parent.parent
REPO = SERVER.parent
sys.path.insert(0, str(SERVER))

from llama_cpp import Llama  # noqa: E402

from chat.context import RAG_NO_HIT_HINT, render_rag_segment  # noqa: E402
from config import config  # noqa: E402
from rag.retriever import search  # noqa: E402
from session_store.sqlite_store import session_store  # noqa: E402

BASE_GGUF = REPO / "Qwen3-14B-Q4_K_M.gguf"
LAW_GGUF = REPO / "law-14b-q4_k_m.gguf"
OUT = SERVER / "spikes" / "law_demo_out.json"
ROLE = "你是一位中国大陆的劳动法咨询助手。"
QUESTION = "我签的是三年期合同，公司说试用期八个月，这样合法吗？"

# 第 ④ 幕：真实法条里没有的规定（虚构条号），塞进知识库后再问
VIRTUAL_DOC = "demo_virtual.txt"
VIRTUAL_TEXT = (
    "《劳动合同法》第一百三十二条：用人单位以电子形式订立劳动合同的，"
    "应当保证劳动者可以随时下载、打印完整合同文本；未提供的，劳动者有权要求补发。\n"
)
VIRTUAL_Q = "公司用电子形式订立劳动合同，但不给我可以下载打印的完整合同文本，可以吗？"


def build_messages(question: str, use_rag: bool, count_tokens) -> tuple[list[dict], list]:
    hits = search(question).hits if use_rag else []
    messages = [{"role": "system", "content": ROLE}]
    if use_rag:
        seg = render_rag_segment(hits, config.rag_max_inject_tokens, count_tokens) \
            if hits else RAG_NO_HIT_HINT
        messages.append({"role": "system", "content": seg})
    messages.append({"role": "user", "content": question})
    return messages, hits


def run_act(title: str, model_path: Path, use_rag: bool, question: str,
            log) -> dict:
    t0 = time.time()
    llm = Llama(model_path=str(model_path), n_ctx=8192, n_gpu_layers=0, verbose=False)
    count = lambda s: len(llm.tokenize(s.encode("utf-8")))
    messages, hits = build_messages(question, use_rag, count)
    out = llm.create_chat_completion(messages=messages, temperature=0,
                                    max_tokens=600)["choices"][0]["message"]["content"]
    body = out.split("</think>")[-1].strip() if "</think>" in out else out.strip()
    result = {
        "act": title, "model": model_path.name, "rag": use_rag,
        "question": question,
        "materials": [h["content"].split("：")[0] for h in hits],
        "answer": body, "seconds": round(time.time() - t0, 1),
    }
    log(f"  {title}: {len(body)} 字 / {result['seconds']}s / 材料 {result['materials']}")
    del llm
    return result


def ingest_virtual(log) -> int:
    for d in session_store.list_documents():
        if d.filename == VIRTUAL_DOC:
            session_store.delete_document(d.id)
    doc = __import__("rag.ingest", fromlist=["ingest_document"]).ingest_document(
        VIRTUAL_DOC, VIRTUAL_TEXT.encode("utf-8"), chunk_mode="article")
    log(f"  已注入虚构条文 #{doc.id}（{doc.chunk_count} 块）")
    return doc.id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()
    started = time.time()
    results = []

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"demo 开始（模型 CPU 推理，一题约 1 分钟）")
    if 1 in args.acts:
        results.append(run_act("① 原版模型 · 无材料（裸考）", BASE_GGUF, False,
                               QUESTION, log))
    if 2 in args.acts:
        results.append(run_act("② 原版模型 · 有材料", BASE_GGUF, True,
                               QUESTION, log))
    if 3 in args.acts:
        results.append(run_act("③ law-14b · 有材料", LAW_GGUF, True,
                               QUESTION, log))
    if 4 in args.acts:
        doc_id = ingest_virtual(log)
        try:
            results.append(run_act("④ law-14b · 材料里是虚构条文", LAW_GGUF, True,
                                   VIRTUAL_Q, log))
        finally:
            session_store.delete_document(doc_id)
            log("  已移除虚构条文，知识库恢复原状")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log(f"完成，总耗时 {round(time.time() - started)}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
