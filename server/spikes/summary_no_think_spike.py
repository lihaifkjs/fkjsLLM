"""Spike 追加：验证关闭 Qwen3 think 的两种方式（压缩调用不需要思考块）。

A. create_chat_completion(chat_template_kwargs={"enable_thinking": False})
B. user 内容尾部加 " /no_think"（Qwen3 模板软开关）

用法:  server/.venv/bin/python server/spikes/summary_no_think_spike.py
"""
import sys
import time
from pathlib import Path

from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "Qwen3-14B-Q4_K_M.gguf")

PROMPT = [{"role": "system", "content": "你是助手。"},
          {"role": "user", "content": "用一句话介绍杭州。"}]


def run(llm, tag, messages, **kwargs):
    t0 = time.perf_counter()
    try:
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=128, temperature=0.2,
            stream=False, **kwargs)
        content = resp["choices"][0]["message"]["content"]
        ms = (time.perf_counter() - t0) * 1000
        has_think = "<think>" in content
        print(f"--- {tag} --- think={has_think}, {ms:.0f}ms, "
              f"completion_tokens={resp['usage']['completion_tokens']}")
        print(repr(content[:120]))
    except TypeError as e:
        print(f"--- {tag} --- TypeError: {e}")


def main():
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=2048,
                verbose=False)
    run(llm, "A chat_template_kwargs", PROMPT,
        chat_template_kwargs={"enable_thinking": False})
    no_think = [dict(PROMPT[0]),
                {"role": "user", "content": PROMPT[1]["content"] + " /no_think"}]
    run(llm, "B /no_think", no_think)
    run(llm, "C 对照（不加任何处理）", PROMPT)
    print("SPIKE_OK")


if __name__ == "__main__":
    main()
