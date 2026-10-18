"""M12.1 批次 0：按条切块的检索召回实测 + 阈值/注入预算重标定。

M7 的 0.40 阈值与 512 token 预算是按「500 字窗口切块」调定的；按条切块
后片段变短（中位数 ~80 字）、互相更相似，分数分布必然移动，必须重测。

输出：top-1/top-3 命中率（22 条口语化问题）、命中/未命中分数区间、
域外 query 最高分（阈值下限依据）、top-3 注入的字符规模（预算依据）。

用法：cd server && python spikes/law_recall_eval.py
"""
import json
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIKE_DIR.parent))

from config import config  # noqa: E402
from rag.chunker import chunk_by_article, is_article_chunk  # noqa: E402
from rag.embedder import embedder  # noqa: E402

CORPUS_DIR = SPIKE_DIR.parent / "rag" / "corpus"
CORPUS_FILES = ["劳动合同法.txt", "劳动合同法实施条例.txt"]
RESULT_PATH = SPIKE_DIR / "law_recall_result.json"
TOP_K = 3

# (口语化问题, 应命中的条文标签) —— 题面不含法条号，避免字面匹配作弊
QUERIES = [
    ("试用期最长能约定多久", "《劳动合同法》第十九条"),
    ("试用期工资不得低于多少", "《劳动合同法》第二十条"),
    ("公司一直不签书面劳动合同，超过一个月怎么赔", "《劳动合同法》第八十二条"),
    ("连续签两次固定期限合同后能不能要求签无固定期限的", "《劳动合同法》第十四条"),
    ("公司违法辞退我能拿多少赔偿金", "《劳动合同法》第八十七条"),
    ("经济补偿按工作年限怎么算", "《劳动合同法》第四十七条"),
    ("员工自己辞职要提前几天通知", "《劳动合同法》第三十七条"),
    ("公司拖欠工资，我可以直接走人吗", "《劳动合同法》第三十八条"),
    ("怀孕期间公司能不能辞退我", "《劳动合同法》第四十二条"),
    ("竞业限制最长能约定几年", "《劳动合同法》第二十四条"),
    ("公司能扣押我的身份证吗", "《劳动合同法》第九条"),
    ("劳务派遣只能在什么岗位上用", "《劳动合同法》第六十六条"),
    ("非全日制用工每天最多工作几个小时", "《劳动合同法》第六十八条"),
    ("公司一次裁掉二十人以上要走什么程序", "《劳动合同法》第四十一条"),
    ("安排加班却不给加班费怎么办", "《劳动合同法》第八十五条"),
    ("员工的劳动合同里必须写哪些内容", "《劳动合同法》第十七条"),
    ("拒绝领导违章指挥算违反劳动合同吗", "《劳动合同法》第三十二条"),
    ("医疗期满不能从事原工作，公司能解除吗", "《劳动合同法》第四十条"),
    ("员工干到退休年龄，劳动合同还有效吗", "《劳动合同法实施条例》第二十一条"),
    ("培训后约定服务期，违约金最多要多少", "《劳动合同法》第二十二条"),
    ("员工给公司造成损失要不要赔", "《劳动合同法》第九十条"),
    ("在我单位连续工作满十年是不是就必须签无固定期限", "《劳动合同法》第十四条"),
]

OFFDOMAIN_QUERIES = [
    "红烧肉怎么做才好吃",
    "2026 年世界杯赛程安排",
    "周杰伦最新专辑叫什么",
    "杭州今天天气怎么样",
    "Vue3 组合式 API 怎么写路由守卫",
    "大模型量化原理是什么",
    "Python 列表推导式语法",
    "明年房价会涨吗",
]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    chunks: list[str] = []
    for name in CORPUS_FILES:
        chunks.extend(chunk_by_article(
            (CORPUS_DIR / name).read_text(encoding="utf-8")))
    assert all(is_article_chunk(c) for c in chunks), "存在非条文 chunk"

    chunk_vecs = embedder.embed_texts(chunks)

    def rank(query: str) -> list[tuple[float, str]]:
        qvec = embedder.embed_texts([config.rag_query_prefix + query])[0]
        scored = [(_cosine(qvec, cv), c) for cv, c in zip(chunk_vecs, chunks)]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:TOP_K]

    rows, hit1, hit3 = [], 0, 0
    for query, expected in QUERIES:
        top = rank(query)
        rank_of = next((i for i, (_, c) in enumerate(top, 1)
                        if c.startswith(expected)), None)
        hit1 += rank_of == 1
        hit3 += rank_of is not None
        rows.append({"query": query, "expected": expected, "rank": rank_of,
                     "top1_score": round(top[0][0], 4),
                     "top_articles": [c.split("\n")[0][:28] for _, c in top]})

    offdomain = [{"query": q, "best_score": round(rank(q)[0][0], 4)}
                 for q in OFFDOMAIN_QUERIES]

    hit_scores = [r["top1_score"] for r in rows if r["rank"] is not None]
    miss_scores = [r["top1_score"] for r in rows if r["rank"] is None]
    injected_chars = []
    for query, expected in QUERIES:
        top = rank(query)
        injected_chars.append(sum(len(c) for _, c in top))

    result = {
        "chunk_count": len(chunks),
        "chunk_char_median": sorted(len(c) for c in chunks)[len(chunks) // 2],
        "hit_at_1": hit1 / len(QUERIES),
        "hit_at_3": hit3 / len(QUERIES),
        "hit_top1_score_min": min(hit_scores) if hit_scores else None,
        "miss_top1_score_max": max(miss_scores) if miss_scores else None,
        "offdomain_max_score": max(o["best_score"] for o in offdomain),
        "inject_top3_chars_max": max(injected_chars),
        "queries": rows,
        "offdomain": offdomain,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"chunk 数 {len(chunks)}，长度中位数 {result['chunk_char_median']} 字")
    print(f"命中率 top-1 {hit1}/{len(QUERIES)}  top-3 {hit3}/{len(QUERIES)}")
    for r in rows:
        flag = "OK " if r["rank"] else "MISS"
        print(f"  {flag} rank={r['rank']} {r['top1_score']:.4f} {r['query']}"
              f"  -> {r['top1_score']:.3f} {r['top_articles'][0]}")
    print(f"\n命中 query 的 top-1 分数下限 {result['hit_top1_score_min']}")
    print(f"未命中 query 的 top-1 分数上限 {result['miss_top1_score_max']}")
    print(f"域外 query 最高分 {result['offdomain_max_score']}")
    print(f"top-3 注入字符数最大 {result['inject_top3_chars_max']}")
    print(f"结果已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
