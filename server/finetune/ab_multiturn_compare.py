"""M8 诊断（一次性）：base Qwen3-14B vs fkjs-14b 对拍。

目的：
  P1 无 /no_think 的推理题 → base 会 think、fkjs 不会？（症状 1 实测）
  P2 无 /no_think 的身份直问 → 同上
  P3 5 轮多轮记忆探针 → 两者是否忘记前文（症状 2 定案）

口径与线上一致：create_chat_completion 默认模板、不传 enable_thinking、
history 里塞模型自己生成的原文（含空 think 壳）。temperature=0 确定性采样。
用法: server/.venv/bin/python3 ab_multiturn_compare.py [--models base fkjs]
"""
import argparse
import re
import sys
import time
from pathlib import Path

from llama_cpp import Llama, llama_supports_gpu_offload

REPO = Path(__file__).resolve().parents[2]
MODELS = {
    "base": REPO / "Qwen3-14B-Q4_K_M.gguf",
    "fkjs": REPO / "fkjs-14b-q4_k_m.gguf",
}
N_CTX = 8192

# P3：先给三个事实，再插 3 轮无关任务，最后追问
MEMORY_TURNS = [
    "请记住三个信息，只回复「已记住」：我的名字是李海，我的猫叫豆豆，"
    "我下周出差的城市是成都。",
    "用一句话解释什么是梯度下降。",
    "讲一个关于海洋的冷知识。",
    "用 Python 写一个读取 JSON 文件的函数，只要代码。",
    "我叫什么？我的猫叫什么？我下周去哪个城市出差？",
]
MEMORY_KEYS = ["李海", "豆豆", "成都"]


def think_split(text: str) -> tuple[str, str]:
    """按 </think> 切成 (think 内容, 正文)；无 think 块时 think 为空。"""
    if "</think>" not in text:
        return "", text
    head, body = text.split("</think>", 1)
    return head.split("<think>")[-1].strip(), body.strip()


def gen(llm: Llama, messages: list[dict], max_tokens: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages, temperature=0.0, top_p=1.0, top_k=1,
        max_tokens=max_tokens, stream=False,
    )
    text = out["choices"][0]["message"]["content"] or ""
    return text, time.perf_counter() - t0


def run_probe(llm: Llama, name: str, tag: str, question: str,
              max_tokens: int) -> str:
    text, dt = gen(llm, [{"role": "user", "content": question}], max_tokens)
    think, body = think_split(text)
    print(f"\n[{name}] {tag} | {dt:.1f}s | think {len(think)} 字 | "
          f"正文 {len(body)} 字")
    print(f"  think: {think[:400] if think else '（空）'}")
    print(f"  正文 : {body[:400]}")
    return text


def run_memory(llm: Llama, name: str) -> None:
    print(f"\n===== [{name}] P3 多轮记忆探针 =====")
    messages: list[dict] = []
    final = ""
    for i, q in enumerate(MEMORY_TURNS, 1):
        messages.append({"role": "user", "content": q})
        text, dt = gen(llm, messages, 192)
        messages.append({"role": "assistant", "content": text})
        think, body = think_split(text)
        print(f"  turn{i} ({dt:.1f}s, think {len(think)} 字): "
              f"{body.strip()[:120].replace(chr(10), ' ')}")
        final = text
    _, body = think_split(final)
    hit = [k for k in MEMORY_KEYS if k in body]
    print(f"  → 末轮命中 {len(hit)}/3 {hit} | 答案：{body.strip()[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "fkjs"])
    args = ap.parse_args()
    gpu = -1 if llama_supports_gpu_offload() else 0
    print(f"n_gpu_layers={gpu} (Metal offload 支持: {llama_supports_gpu_offload()})")
    for name in args.models:
        path = MODELS[name]
        print(f"\n===== 加载 {name}: {path.name} =====", flush=True)
        t0 = time.perf_counter()
        llm = Llama(model_path=str(path), n_ctx=N_CTX, n_gpu_layers=gpu,
                    verbose=False)
        print(f"  加载 {time.perf_counter() - t0:.1f}s", flush=True)
        run_probe(llm, name, "P1 推理题（无 /no_think）",
                  "9.11 和 9.9 哪个大？", 768)
        run_probe(llm, name, "P2 身份直问（无 /no_think）", "你是谁？", 256)
        run_memory(llm, name)
        del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
