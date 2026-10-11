"""M7 批次 0 补充：阈值下限取证 + 对拍差异诊断。

主 spike（m7_embedding_spike.py）发现：
1. 命中/未命中分数有交叠，阈值的真实作用是拦「与知识库无关的 query」——
   需要域外 query 的分数分布来定阈值下限。
2. 两栈对拍余弦 0.93-0.95，未到 ≈1——检查两侧原始向量范数，定位差异来源。
"""

import json
import time
from pathlib import Path

from m7_embedding_spike import (
    DOCS,
    MODELS_DIR,
    QUERY_PREFIX,
    StackA,
    StackB,
    chunk_text,
    cosine,
    l2_normalize,
)

OFFDOMAIN_QUERIES = [
    "红烧肉怎么做才好吃",
    "2026 年世界杯赛程安排",
    "周杰伦最新专辑叫什么",
    "杭州今天天气怎么样",
]

RESULT_PATH = Path(__file__).resolve().parent / "m7_embedding_result_supplement.json"


def raw_norms(stack_name: str, embed_fn) -> dict:
    norms = [round(float(sum(float(v) * float(v) for v in vec)) ** 0.5, 4) for vec in embed_fn]
    return {"stack": stack_name, "raw_norms": norms}


def main() -> None:
    probe_texts = ["你好", "SQLite 是嵌入式数据库。", "为这个句子生成表示以用于检索相关文章：你好"]
    result: dict = {}

    stack_a = StackA(MODELS_DIR / "bge-small-zh-v1.5-f16.gguf")
    stack_b = StackB()

    # 原始向量范数（不归一化）
    result["raw_norms"] = [
        raw_norms(stack_a.name, stack_a.llm.embed(probe_texts)),
        raw_norms(stack_b.name, [list(v) for v in stack_b.model.embed(probe_texts)]),
    ]

    # 域外 query 分数（chunk_size=500，与主 spike 推荐档一致）
    chunks: list[tuple[int, str]] = []
    for i, doc in enumerate(DOCS):
        for c in chunk_text(doc["content"], 500, 50):
            chunks.append((i, c))

    for stack in (stack_a, stack_b):
        entry = {}
        chunk_vecs = stack.embed([c for _, c in chunks])
        for use_prefix in (True, False):
            t0 = time.perf_counter()
            q_vecs = stack.embed([QUERY_PREFIX + q if use_prefix else q for q in OFFDOMAIN_QUERIES])
            embed_ms = (time.perf_counter() - t0) / len(OFFDOMAIN_QUERIES) * 1000
            scores = []
            for q, qv in zip(OFFDOMAIN_QUERIES, q_vecs):
                best = max(cosine(qv, cv) for cv in chunk_vecs)
                scores.append({"query": q, "best_score": round(best, 4)})
            entry[f"prefix={use_prefix}"] = {
                "offdomain_scores": scores,
                "max_offdomain_score": max(s["best_score"] for s in scores),
                "avg_query_embed_ms": round(embed_ms, 1),
            }
        result[stack.name] = entry

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("原始向量范数（未归一化）:")
    for rn in result["raw_norms"]:
        print(f"  {rn['stack']}: {rn['raw_norms']}")
    print("\n域外 query 最高分（chunk_size=500）:")
    for name in (stack_a.name, stack_b.name):
        for k, e in result[name].items():
            print(f"  {name} {k}: max={e['max_offdomain_score']}"
                  f"  明细={[(s['query'][:6], s['best_score']) for s in e['offdomain_scores']]}")
    print(f"\n结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
