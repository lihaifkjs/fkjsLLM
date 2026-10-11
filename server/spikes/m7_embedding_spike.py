"""M7 批次 0：embedding 推理栈选型 + 切块/检索调优 spike。

对比候选 A（llama-cpp-python 加载 bge-small-zh-v1.5 GGUF）与
候选 B（fastembed / onnxruntime），网格维度：
  栈 × query 前缀有无 × chunk_size ∈ {300, 500, 800}（overlap 固定 50）
统计 Top-3/Top-1 命中率、命中/未命中分数间隔、embedding 耗时、加载耗时，
并做两栈同文本向量对拍（余弦应 ≈1）。结果写 m7_embedding_result.json。
"""

import json
import math
import os
import time
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SPIKE_DIR.parent / "models"
RESULT_PATH = SPIKE_DIR / "m7_embedding_result.json"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
CHUNK_SIZES = [300, 500, 800]
CHUNK_OVERLAP = 50
TOP_K = 3

# ---------------------------------------------------------------- 语料
DOCS = [
    {
        "name": "llm_inference.md",
        "content": (
            "大模型推理优化工程实践\n\n"
            "在 CPU 上部署大语言模型时，量化是最有效的加速手段之一。GGUF 格式支持 Q4_K_M、Q8_0 等多种量化档位，"
            "Q4_K_M 在几乎不损失精度的情况下将 14B 模型压缩到 9GB 以内，使消费级硬件也能运行。\n\n"
            "KV cache 是推理内存占用的大头。每生成一个 token，注意力层都要缓存 Key 和 Value 矩阵，"
            "上下文越长占用越大。llama.cpp 支持对 KV cache 也做量化（如 q8_0），可以显著降低长上下文场景的内存压力。\n\n"
            "n_gpu_layers 参数控制有多少层卸载到 GPU。显存不足时只卸载部分层，其余留在 CPU 计算，"
            "速度介于纯 CPU 与纯 GPU 之间。批大小 n_batch 影响 prompt 处理吞吐，但不影响逐 token 生成速度。\n\n"
            "连续批处理（continuous batching）是服务端提升吞吐的关键技术，vLLM 的 PagedAttention "
            "借鉴操作系统虚拟内存思想管理 KV cache 显存，把显存碎片率降到 4% 以下。"
        ),
    },
    {
        "name": "vue3_frontend.md",
        "content": (
            "Vue 3 前端开发笔记\n\n"
            "Vue 3 的组合式 API 让逻辑复用变得简单。useMouse、useFetch 这类 composable 函数可以把响应式状态"
            "和副作用封装起来，在多个组件间共享，比 Vue 2 的 mixin 更清晰，也不会有命名冲突问题。\n\n"
            "Pinia 是官方推荐的状态管理库。defineStore 定义 store，state 用 ref 或 reactive，"
            "getters 用 computed，actions 就是普通方法。与 Vuex 相比没有了 mutation，类型推导也更友好。\n\n"
            "响应式原理上，Vue 3 用 Proxy 替代了 Object.defineProperty，可以监听数组下标变化和对象新增属性，"
            "不再需要 Vue.set。ref 包装基本类型，reactive 处理对象，模板中 ref 自动解包。\n\n"
            "Vite 利用浏览器原生 ESM 实现按需编译，开发服务器冷启动是毫秒级的，"
            "生产构建则用 Rollup 打包。环境变量以 VITE_ 前缀暴露给客户端代码。"
        ),
    },
    {
        "name": "sqlite_notes.md",
        "content": (
            "SQLite 使用经验\n\n"
            "SQLite 是嵌入式数据库，整个库就是一个文件，非常适合桌面应用和小型服务。"
            "WAL 模式（Write-Ahead Logging）允许多个读者与一个写者并发，读操作不再阻塞写操作，"
            "开启方式是执行 PRAGMA journal_mode=WAL。\n\n"
            "索引能大幅加速查询，但不是越多越好。每个索引都会拖慢 INSERT 和 UPDATE，"
            "因为写操作要同步维护索引树。复合索引遵循最左前缀原则：对 (a, b) 建索引可以服务 a 的查询，"
            "但不能只服务 b 的查询。\n\n"
            "PRAGMA user_version 是一个用户自定义的整数字段，常被用来实现 schema 迁移："
            "应用启动时读取当前版本号，按版本顺序执行增量迁移脚本，保证旧库文件能平滑升级。\n\n"
            "SQLite 的 BLOB 类型可以存任意字节流，比如把 float32 数组按小端序打包后存进去，"
            "读取时再用 struct 或 array 模块解包，不需要引入额外依赖。"
        ),
    },
]

# (query, 应命中的文档下标)
QUERIES = [
    ("Q4_K_M 量化能把模型压到多大", 0),
    ("KV cache 量化怎么省内存", 0),
    ("显存不够时 n_gpu_layers 怎么设置", 0),
    ("vLLM 是怎么管理显存碎片的", 0),
    ("Pinia 和 Vuex 有什么区别", 1),
    ("Vue3 为什么不再需要 Vue.set", 1),
    ("Vite 开发服务器为什么启动快", 1),
    ("WAL 模式对并发读写有什么帮助", 2),
    ("复合索引的最左前缀原则是什么", 2),
    ("怎么用 user_version 做数据库迁移", 2),
]

# 对拍固定文本
PARITY_TEXTS = [
    "为这个句子生成表示以用于检索相关文章：KV cache 量化怎么省内存",
    "KV cache 是推理内存占用的大头。每生成一个 token，注意力层都要缓存 Key 和 Value 矩阵。",
    "SQLite 是嵌入式数据库，整个库就是一个文件。",
]


# ---------------------------------------------------------------- 切块
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """固定窗口 + 重叠切块，优先在窗口内最后一个段落边界（\\n\\n）处切。"""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            boundary = text.rfind("\n\n", start + overlap, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1) if end < n else n
    return chunks


def l2_normalize(vec) -> list[float]:
    vec = [float(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


# ---------------------------------------------------------------- 栈封装
class StackA:  # llama-cpp-python + GGUF
    name = "A_llamacpp_gguf"

    def __init__(self, model_path: Path):
        from llama_cpp import Llama

        t0 = time.perf_counter()
        self.llm = Llama(model_path=str(model_path), embedding=True, n_ctx=512, verbose=False)
        self.load_seconds = time.perf_counter() - t0
        try:
            self.pooling_type = self.llm.pooling_type()
        except Exception:
            self.pooling_type = "unknown"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize(v) for v in self.llm.embed(texts)]


class StackB:  # fastembed / onnxruntime
    name = "B_fastembed_onnx"

    def __init__(self):
        from fastembed import TextEmbedding

        t0 = time.perf_counter()
        self.model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        self.load_seconds = time.perf_counter() - t0
        self.pooling_type = "cls(onnx)"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize(list(v)) for v in self.model.embed(texts)]


# ---------------------------------------------------------------- 评测
def evaluate(stack, chunk_size: int, use_prefix: bool) -> dict:
    doc_chunks: list[tuple[int, str]] = []  # (doc_idx, chunk)
    for i, doc in enumerate(DOCS):
        for c in chunk_text(doc["content"], chunk_size, CHUNK_OVERLAP):
            doc_chunks.append((i, c))

    t0 = time.perf_counter()
    chunk_vecs = stack.embed([c for _, c in doc_chunks])
    doc_embed_ms = (time.perf_counter() - t0) / len(doc_chunks) * 1000

    queries = [(QUERY_PREFIX + q if use_prefix else q, doc) for q, doc in QUERIES]
    t0 = time.perf_counter()
    query_vecs = stack.embed([q for q, _ in queries])
    query_embed_ms = (time.perf_counter() - t0) / len(queries) * 1000

    top1_hits = topk_hits = 0
    hit_scores, miss_scores = [], []
    details = []
    for (q_text, expected_doc), qv in zip(queries, query_vecs):
        scored = sorted(
            ((cosine(qv, cv), doc_idx) for (doc_idx, _), cv in zip(doc_chunks, chunk_vecs)),
            reverse=True,
        )
        top = scored[:TOP_K]
        if top[0][1] == expected_doc:
            top1_hits += 1
        if any(d == expected_doc for _, d in top):
            topk_hits += 1
        best_hit = max(s for s, d in scored if d == expected_doc)
        best_miss = max(s for s, d in scored if d != expected_doc)
        hit_scores.append(best_hit)
        miss_scores.append(best_miss)
        details.append({
            "query": q_text, "expected_doc": expected_doc,
            "top3": [{"score": round(s, 4), "doc": d} for s, d in top],
            "best_hit_score": round(best_hit, 4), "best_miss_score": round(best_miss, 4),
        })

    n = len(queries)
    margin = min(h - m for h, m in zip(hit_scores, miss_scores))
    return {
        "chunk_size": chunk_size, "query_prefix": use_prefix,
        "n_chunks": len(doc_chunks),
        "top1_hit_rate": top1_hits / n, "top3_hit_rate": topk_hits / n,
        "min_hit_score": round(min(hit_scores), 4),
        "max_miss_score": round(max(miss_scores), 4),
        "min_margin": round(margin, 4),
        "avg_doc_embed_ms": round(doc_embed_ms, 1),
        "avg_query_embed_ms": round(query_embed_ms, 1),
        "queries": details,
    }


def main() -> None:
    result: dict = {"stacks": {}, "parity": None, "notes": []}

    stacks = []
    gguf = MODELS_DIR / "bge-small-zh-v1.5-f16.gguf"
    if gguf.exists():
        try:
            stacks.append(StackA(gguf))
        except Exception as e:  # noqa: BLE001
            result["notes"].append(f"栈 A 加载失败: {e!r}")
    else:
        result["notes"].append(f"栈 A 模型文件不存在: {gguf}")
    try:
        stacks.append(StackB())
    except Exception as e:  # noqa: BLE001
        result["notes"].append(f"栈 B 加载失败: {e!r}")

    for stack in stacks:
        entry = {
            "load_seconds": round(stack.load_seconds, 2),
            "pooling_type": stack.pooling_type,
            "runs": [],
        }
        for size in CHUNK_SIZES:
            for prefix in (True, False):
                entry["runs"].append(evaluate(stack, size, prefix))
        result["stacks"][stack.name] = entry

    if len(stacks) == 2:
        parity = []
        for text in PARITY_TEXTS:
            va, vb = stacks[0].embed([text])[0], stacks[1].embed([text])[0]
            parity.append({"text": text[:30] + "...", "cosine": round(cosine(va, vb), 4)})
        result["parity"] = parity

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # ---------------- 摘要表
    print(f"\n{'栈':<20}{'加载(s)':>8}  pooling")
    for name, e in result["stacks"].items():
        print(f"{name:<20}{e['load_seconds']:>8}  {e['pooling_type']}")
    print(f"\n{'栈':<20}{'size':>5}{'前缀':>5}{'块数':>5}{'Top1':>7}{'Top3':>7}"
          f"{'min命中':>9}{'max未中':>9}{'min间隔':>9}{'doc ms':>8}{'qry ms':>8}")
    for name, e in result["stacks"].items():
        for r in e["runs"]:
            print(f"{name:<20}{r['chunk_size']:>5}{str(r['query_prefix']):>5}{r['n_chunks']:>5}"
                  f"{r['top1_hit_rate']:>7.1%}{r['top3_hit_rate']:>7.1%}"
                  f"{r['min_hit_score']:>9}{r['max_miss_score']:>9}{r['min_margin']:>9}"
                  f"{r['avg_doc_embed_ms']:>8}{r['avg_query_embed_ms']:>8}")
    if result["parity"]:
        print("\n两栈对拍（同文本向量余弦）:")
        for p in result["parity"]:
            print(f"  {p['cosine']}  {p['text']}")
    for note in result["notes"]:
        print(f"NOTE: {note}")
    print(f"\n结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
