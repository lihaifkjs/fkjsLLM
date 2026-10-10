"""Spike: 验证 M5 观测采集路径（prompt 渲染复现 / 精确 usage / perf 计时）。

对拍目标：
1. prompt_render 复现的渲染文本 token 数 == create_chat_completion 非流式
   返回的 usage.prompt_tokens（证明渲染与 llama-cpp 内部逐字节一致）
2. segment_tokens 三段之和 == input_tokens
3. reset_timings + llama_perf_context：n_eval == usage.completion_tokens，
   t_p_eval_ms / t_eval_ms > 0
4. 同上下文第二轮：n_p_eval < 第一轮 prompt_tokens（KV 前缀复用生效）

用法:  server/.venv/bin/python server/spikes/m5_observability_spike.py
"""

import sys
from pathlib import Path

from llama_cpp import Llama, llama_cpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_manager import prompt_render  # noqa: E402

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "Qwen3-14B-Q4_K_M.gguf")

MESSAGES_1 = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "用一句话介绍杭州。"},
]
MESSAGES_2 = MESSAGES_1 + [
    {"role": "assistant",
     "content": "<think>\n用户问杭州，一句话即可。\n</think>\n\n杭州是浙江省省会，以西湖闻名。"},
    {"role": "user", "content": "那西湖十景有哪些？"},
]


def main():
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=2048,
                verbose=False)
    print(f"chat_format = {llm.chat_format}")
    assert llm.chat_format == "chat_template.default", \
        "渲染路径前提不成立：实际未走 Jinja2ChatFormatter"

    for round_no, messages in enumerate((MESSAGES_1, MESSAGES_2), 1):
        rendered = prompt_render.render_prompt(llm, messages)
        input_tokens = prompt_render.count_text(llm, rendered)
        seg = prompt_render.segment_tokens(llm, messages)

        llm._ctx.reset_timings()
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=8, stream=False)
        perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
        usage = resp["usage"]

        print(f"\n--- 第 {round_no} 轮 ---")
        print(f"渲染文本前 60 字符: {rendered[:60]!r}")
        print(f"渲染文本后 40 字符: {rendered[-40:]!r}")
        print(f"账单口径 input_tokens = {input_tokens},"
              f" usage.prompt_tokens = {usage['prompt_tokens']}")
        print(f"seg_tokens = {seg}, 三段和 = {sum(seg.values())}")
        print(f"n_p_eval = {perf.n_p_eval}, n_eval = {perf.n_eval},"
              f" usage.completion_tokens = {usage['completion_tokens']}")
        print(f"t_p_eval_ms = {perf.t_p_eval_ms:.1f},"
              f" t_eval_ms = {perf.t_eval_ms:.1f}")

        assert input_tokens == usage["prompt_tokens"], "渲染对拍失败"
        assert sum(seg.values()) == input_tokens, "分段之和不等于总输入"
        assert all(v >= 0 for v in seg.values()), f"分段出现负值: {seg}"
        assert perf.n_eval + 1 == usage["completion_tokens"], \
            "n_eval+1 对拍失败（首 token 来自 prefill logits，不占 decode 步）"
        assert perf.t_p_eval_ms > 0 and perf.t_eval_ms > 0
        if round_no == 2:
            assert perf.n_p_eval < usage["prompt_tokens"], \
                "第二轮未见 KV 前缀复用"

    print("\nSPIKE_OK")


if __name__ == "__main__":
    main()
