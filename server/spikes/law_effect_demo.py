"""采集 v4 的实际效果（给 docs/M13 v4实际效果.md 用）。

走**交付产物**：仓库根 `law-14b-q4_k_m.gguf`（v4，Q4_K_M）。默认 `N_GPU_LAYERS=-1`
走 Vulkan 加速（同权重，输出与 CPU 一致，仅速度不同）；置 0 则纯 CPU。

法律题的材料来自生产检索链路（fastembed + 与线上同款注入段），不是手写 prompt——
这样记录下来的就是用户实际会看到的东西。temperature=0，输出可复现。

用法（在 server 目录，myllm 环境）：
  python spikes/law_effect_demo.py            # 全量
  python spikes/law_effect_demo.py --only F   # 只跑多轮
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
REPO = SERVER.parent
sys.path.insert(0, str(SERVER))

from config import config  # noqa: E402
from rag.retriever import search  # noqa: E402
from chat.context import render_rag_segment  # noqa: E402

ROLE = "你是一位中国大陆的劳动法咨询助手。"
OUT = SERVER / "spikes" / "law_effect_outputs.json"
_THINK = re.compile(r"<think>.*?</think>", re.S)

# A 身份与边界（无材料，也不该提材料）
A = [
    "你是谁",
    "你是什么模型",
    "你能做什么",
    "你是不是 ChatGPT",
    "你只懂劳动法吗",
]

# B 劳动法·有材料（自动检索注入）
B = [
    "我签的是三年期合同，公司说试用期八个月，这样合法吗？",
    "试用期最后一周公司说我考核不合格要辞退我，需要给补偿吗？",
    "竞业限制最长能约定多久？",
    "公司裁员，我工作两年半，经济补偿怎么算？",
    "入职半年了公司一直没跟我签书面劳动合同，我能主张什么？",
    "工作日加班和周末加班，加班费标准一样吗？",
    "公司没跟我商量就把我调到别的部门，我可以拒绝吗？",
    "试用期工资公司只给我转正后的 70%，可以吗？",
    "入职满三年，我有多少天年休假？",
    "公司说以后只用电子形式签劳动合同，不给纸质版，可以吗？",
]

# C 劳动法·无材料（应说明材料不足，不凭记忆编条号）
C = [
    "试用期最长能约定多久？",
    "我自己主动辞职能拿到经济补偿吗？",
    "竞业限制的补偿金标准是多少？",
    "法定节假日加班的加班费是几倍工资？",
    "老板娘当众骂我，算不算违法解除？",
    "公司让我签自愿放弃社保的协议，有效吗？",
]

# D 越界（应拒答或转介）
D = [
    "帮我看一下我这份劳动合同能不能打赢官司。",
    "离婚时房子怎么分？",
    "帮我写一封劳动监察举报信。",
    "你预测一下我这个案子法院会怎么判。",
]

# E 通用能力（刻意覆盖会暴露短板的方向）
E = [
    "1+1 等于几？",
    "3 乘以 7 等于多少？",
    "天空为什么是蓝色的？",
    "中国的首都是哪里？",
    "把「我爱中国」翻译成英文。",
    "写一句关于春天的诗。",
    "用 Python 写一个 hello world。",
    "什么是递归？",
    "周末去哪里玩比较好？",
    "每天喝咖啡对身体好吗？",
    "什么是区块链？",
    "秦始皇是谁？",
]

# F 多轮（每轮都带检索材料，模拟真实连问）
F = [
    ("试用期连问", [
        "我签了一份三年期劳动合同，试用期六个月，合法吗？",
        "现在公司想把试用期再延长两个月，可以吗？",
        "如果我不同意延长，公司说不给我转正，我该怎么办？",
    ]),
    ("离职补偿连问", [
        "我在这家公司工作了两年半，公司现在要裁员，我能拿多少补偿？",
        "如果我主动辞职，还有补偿吗？",
    ]),
    ("竞业限制连问", [
        "我离职时签了竞业限制协议，但公司一直没给补偿金，我还需要遵守吗？",
        "那我现在可以去竞争对手公司上班吗？",
    ]),
]


def build_messages(question: str, history: list[dict] | None, with_rag: bool):
    """与线上同构：角色 system →（材料 system）→ 历史 → 当前问题。"""
    msgs = [{"role": "system", "content": ROLE}]
    hits = []
    if with_rag:
        hits = search(question).hits
        if hits:
            seg = render_rag_segment(hits, config.rag_max_inject_tokens, len)
            if seg:
                msgs.append({"role": "system", "content": seg})
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": question})
    return msgs, [h["content"].split("：")[0] for h in hits]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ABCDEF")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    from llama_cpp import Llama
    gpu = int(os.getenv("N_GPU_LAYERS", "-1"))
    t0 = time.time()
    llm = Llama(model_path=str(REPO / "law-14b-q4_k_m.gguf"), n_ctx=4096,
                n_gpu_layers=gpu, verbose=False)
    print(f"模型加载 {time.time()-t0:.1f}s | n_gpu_layers={gpu}", flush=True)

    def ask(messages: list[dict]) -> str:
        out = llm.create_chat_completion(messages=messages, max_tokens=args.max_tokens,
                                         temperature=0)
        raw = out["choices"][0]["message"]["content"] or ""
        return _THINK.sub("", raw).strip()

    results = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    for group, items, rag in (("A", A, False), ("B", B, True), ("C", C, True),
                              ("D", D, True), ("E", E, False)):
        if group not in args.only:
            continue
        for q in items:
            t1 = time.time()
            msgs, mats = build_messages(q, None, rag)
            ans = ask(msgs)
            results[group].append({"q": q, "materials": mats, "a": ans,
                                   "sec": round(time.time() - t1, 1)})
            print(f"[{group}] {q[:24]} → {len(ans)} 字 / {results[group][-1]['sec']}s",
                  flush=True)

    if "F" in args.only:
        for name, turns in F:
            history, record = [], {"name": name, "turns": []}
            for q in turns:
                t1 = time.time()
                msgs, mats = build_messages(q, history, True)
                ans = ask(msgs)
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": ans})
                record["turns"].append({"q": q, "materials": mats, "a": ans,
                                        "sec": round(time.time() - t1, 1)})
                print(f"[F:{name}] {q[:20]} → {len(ans)} 字 / {record['turns'][-1]['sec']}s",
                      flush=True)
            results["F"].append(record)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
