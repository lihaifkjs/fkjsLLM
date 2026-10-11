"""Spike: M6.5 摘要压缩调优（批次 0）。

目标：
1. 验证 14B 用压缩模板产出的摘要质量（人工读：关键事实保留度）
2. 测量单次压缩耗时（prefill/decode 分段）与摘要 token 数分布
3. 对比 temperature 0.2 / 0.7 的稳定性，对比全量 vs 增量两步压缩
4. 为 SUMMARY_KEEP_ROUNDS / SUMMARY_MAX_TOKENS / SUMMARY_TEMPERATURE 定值提供数据

用法:  server/.venv/bin/python server/spikes/summary_compression_spike.py
产出:  server/spikes/summary_compression_result.json
"""

import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama, llama_cpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_manager import prompt_render  # noqa: E402

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "Qwen3-14B-Q4_K_M.gguf")
RESULT_PATH = Path(__file__).with_name("summary_compression_result.json")

SUMMARY_SYSTEM = (
    "你是对话压缩器。将【既有摘要】与【新增对话】合并为一份连贯摘要，"
    "保留关键事实、用户偏好、结论与待办，丢弃寒暄与细节复述。"
)
SUMMARY_USER_TMPL = (
    "【既有摘要】\n{prev_summary}\n\n"
    "【新增对话】\n{source}\n\n"
    "【要求】仅输出合并后的摘要，不超过 {max_chars} 字。"
)

# 10 轮构造对话，埋入可核查事实：名字/偏好/决策/待办/数字
ROUNDS = [
    ("我叫李雷，以后请用这个名字称呼我。",
     "好的李雷，已记住你的名字，后续对话会这样称呼你。"),
    ("我在做一个局域网 LLM 项目，技术栈是 Vue3 + FastAPI + llama-cpp-python。",
     "了解，前端 Vue3、后端 FastAPI、推理用 llama-cpp-python，部署在局域网内。"),
    ("模型定的是 Qwen3-14B 的 Q4_K_M 量化版，显存不够先跑 CPU。",
     "明白，Qwen3-14B-Q4_K_M，当前 CPU 推理，后续可切换 CUDA 加速。"),
    ["帮我列一下二期要做的三件事，我记不住。",
     "二期三件事：1. 可观测性面板；2. 参数实验场；3. 上下文摘要压缩。"],
    ("对了，deadline 是 10 月 31 号，这天之前必须交付。",
     "已记录：交付截止日为 10 月 31 日。"),
    ("上下文的 n_ctx 是 8192，滑窗从最早的用户助手对开始丢。",
     "确认：n_ctx=8192，滑窗按 user/assistant 成对从最早开始丢弃。"),
    ("我决定数据库用 SQLite，迁移走 PRAGMA user_version。",
     "好的，SQLite 落库，schema 迁移用 PRAGMA user_version 机制。"),
    ("中午吃的什么推荐给同事？要辣一点的。",
     "推荐麻辣香锅或水煮鱼，辣度可选，适合多人分享。"),
    ("压缩策略里摘要要保留哪些东西来着？提醒我一下。",
     "摘要应保留：关键事实、用户偏好、结论与待办；丢弃寒暄与细节复述。"),
    ("好，那下一步先写 spike 验证压缩质量，你确认下这个顺序对不对。",
     "顺序正确：先 spike 实测压缩质量与耗时定常量，再做后端与前端实现。"),
]

KEY_FACTS = ["李雷", "10 月 31", "8192", "SQLite", "user_version",
             "Vue3", "FastAPI", "Qwen3"]


def render_source(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def compress(llm, prev_summary: str, new_messages: list[dict],
             temperature: float, max_tokens: int) -> dict:
    user = SUMMARY_USER_TMPL.format(
        prev_summary=prev_summary or "（无）",
        source=render_source(new_messages),
        max_chars=400,
    )
    messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user}]
    source_tokens = sum(prompt_render.count_text(llm, m["content"])
                        for m in new_messages)

    llm._ctx.reset_timings()
    t0 = time.perf_counter()
    try:
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=False,
            chat_template_kwargs={"enable_thinking": False})
    except TypeError:  # 模板不支持 enable_thinking 时回退
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=False)
    wall_ms = (time.perf_counter() - t0) * 1000
    perf = llama_cpp.llama_perf_context(llm._ctx.ctx)

    summary = resp["choices"][0]["message"]["content"].strip()
    return {
        "summary": summary,
        "source_tokens": source_tokens,
        "summary_tokens": prompt_render.count_text(llm, summary),
        "wall_ms": round(wall_ms, 1),
        "ttft_ms": round(perf.t_p_eval_ms, 1),
        "decode_ms": round(perf.t_eval_ms, 1),
        "key_facts_hit": [f for f in KEY_FACTS if f in summary],
    }


def main():
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=4096,
                verbose=False)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": c}
               for i, pair in enumerate(ROUNDS) for c in pair]
    results = {}

    # A/B：全量压缩，temperature 对比
    for tag, temp in (("full_t0.2", 0.2), ("full_t0.7", 0.7)):
        results[tag] = compress(llm, "", history, temp, 512)
        r = results[tag]
        print(f"--- {tag} ---\n{r['summary']}\n"
              f"tokens {r['source_tokens']}→{r['summary_tokens']},"
              f" wall {r['wall_ms']:.0f}ms, facts {r['key_facts_hit']}\n")

    # C/D：增量两步（前 5 轮 → 摘要；摘要 + 后 5 轮 → 最终摘要）
    results["incr_step1"] = compress(llm, "", history[:10], 0.2, 512)
    results["incr_step2"] = compress(
        llm, results["incr_step1"]["summary"], history[10:], 0.2, 512)
    for tag in ("incr_step1", "incr_step2"):
        r = results[tag]
        print(f"--- {tag} ---\n{r['summary']}\n"
              f"tokens {r['source_tokens']}→{r['summary_tokens']},"
              f" wall {r['wall_ms']:.0f}ms, facts {r['key_facts_hit']}\n")

    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已写入 {RESULT_PATH}")
    print("SPIKE_OK")


if __name__ == "__main__":
    main()
