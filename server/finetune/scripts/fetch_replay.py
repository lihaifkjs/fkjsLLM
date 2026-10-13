"""批次 1：replay 数据构造（防灾难性遗忘）。

两个来源：
  belle —— 开源中文指令集抽样（通用指令 replay）
  tool  —— 程序化合成 tool calling replay（与 spikes/tool_calling_baseline.json 同分布：
          同 web_search 工具、同 system prompt 格式、同 <tool_call> 输出格式）

用法:
  .../python fetch_replay.py belle [--n 800]
  .../python fetch_replay.py tool  [--n-call 90 --n-nocall 60]
"""

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_FINETUNE = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_FINETUNE / "data" / "raw"
BELLE_DIR = REPO_FINETUNE / "data" / "external" / "belle"
BELLE_FILE = BELLE_DIR / "Belle_open_source_0.5M.jsonl"
BELLE_DATASET = "AI-ModelScope/train_0.5M_CN"

# 身份冲突过滤：replay 里出现这些说法会污染 fkjs 身份，直接丢弃
IDENTITY_CONFLICT_RE = re.compile(
    r"作为(一个|一款)?(AI|人工智能|AI语言模型|语言模型)|"
    r"我是(ChatGPT|通义千问|文心一言|Kimi)|OpenAI|作为AI"
)

# 与 spikes/tool_calling_spike.py 保持一致的工具定义与 system prompt 格式
TOOL_SYSTEM_TMPL = "当前日期 {date}，你可以使用 web_search 工具获取实时信息。 /no_think"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取实时信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    }
]
TOOL_CALL_TMPL = '<tool_call>\n{{"name": "web_search", "arguments": {{"query": "{q}"}}}}\n</tool_call>'

DEFAULT_MODEL = REPO_FINETUNE / "models" / "Qwen3-14B-4bit"


def render_tools_block(model_path: Path) -> str:
    """提取 Qwen3 模板渲染 tools 定义时插入 system 区块的文本，

    使 replay_tool 样本的 system 与线上 create_chat_completion(tools=...) 渲染一致。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    probe = tok.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": "x"}],
        tools=TOOLS, tokenize=False,
    )
    return probe.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆", "东京", "纽约"]
PRODUCTS = ["iPhone", "MacBook", "小米手环", "华为 Mate", "PS5", "Switch", "大疆无人机", "特斯拉 Model Y"]
EVENTS = ["世界杯", "NBA 总决赛", "欧冠", "诺贝尔文学奖", "奥斯卡最佳影片", "亚运会"]
TOPICS = ["科技", "AI", "体育", "财经", "游戏", "汽车", "航天", "半导体"]
SOFTWARE = ["llama.cpp", "Python", "Vue", "Linux 内核", "Chrome", "PyTorch", "Rust", "Node.js"]
CURRENCIES = ["美元", "欧元", "日元", "英镑", "韩元"]

CALL_QUERY_TMPLS = [
    "今天{c}的天气怎么样", "{c}现在气温多少度", "今天{c}会下雨吗", "这周末{c}天气如何",
    "最新一代{p}的价格是多少", "{p}最新款什么时候发布", "现在买{p}要多少钱",
    "{y}年{e}的冠军是谁", "{y}年{e}结果出来了吗",
    "帮我查一下最近的{t}新闻", "最近{t}领域有什么新进展", "今天的{t}热搜有哪些",
    "搜索一下{s}最新版本号", "{s}最近更新了什么",
    "现在{cur}兑人民币的汇率是多少", "今天{cur}汇率多少",
    "今天A股收盘怎么样", "最近有什么新上映的电影", "现在油价多少钱一升",
]

# query 参数取问题的精简版（与基线实测口径一致：去掉祈使前缀，保留核心实体）
STRIP_PREFIX_RE = re.compile(r"^(帮我|请|麻烦)?(查一下|查下|搜索一下|搜一下|查查|检索一下)")


def tool_system(rng: random.Random, tools_block: str) -> str:
    d = date(2026, 1, 1) + timedelta(days=rng.randrange(365))
    return TOOL_SYSTEM_TMPL.format(date=d.isoformat()) + tools_block


def reservoir_sample(path: Path, n: int, rng: random.Random,
                     max_q: int = 300, max_a: int = 600) -> list[dict]:
    """水库抽样，避免把 0.5M 行全读进内存之外的额外代价（文件本身仍顺序扫一遍）。"""
    sample = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (row.get("instruction") or "").strip()
            a = (row.get("output") or "").strip()
            if not q or not a or len(q) > max_q or len(a) > max_a:
                continue
            if IDENTITY_CONFLICT_RE.search(q) or IDENTITY_CONFLICT_RE.search(a):
                continue
            if len(sample) < n:
                sample.append({"q": q, "a": a})
            else:
                j = rng.randrange(i + 1)
                if j < n:
                    sample[j] = {"q": q, "a": a}
    return sample


def cmd_belle(args):
    if not BELLE_FILE.exists():
        BELLE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"下载 Belle 数据集 → {BELLE_DIR}")
        subprocess.run([
            str(Path(sys.executable).parent / "modelscope"), "download",
            "--dataset", BELLE_DATASET, "Belle_open_source_0.5M.jsonl",
            "--local_dir", str(BELLE_DIR),
        ], check=True)
    rng = random.Random(args.seed)
    rows = reservoir_sample(BELLE_FILE, args.n, rng)
    out = RAW_DIR / "replay_general.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "category": "replay_general", "system": None,
                "question": r["q"], "answer": r["a"], "source": "belle", "seed": None,
            }, ensure_ascii=False) + "\n")
    print(f"[replay_general] 写入 {len(rows)} 条 → {out}")


def cmd_tool(args):
    rng = random.Random(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tools_block = render_tools_block(Path(args.model))
    records = []

    for _ in range(args.n_call):
        tmpl = rng.choice(CALL_QUERY_TMPLS)
        q = tmpl.format(
            c=rng.choice(CITIES), p=rng.choice(PRODUCTS), y=rng.choice([2025, 2026]),
            e=rng.choice(EVENTS), t=rng.choice(TOPICS), s=rng.choice(SOFTWARE),
            cur=rng.choice(CURRENCIES),
        )
        query_arg = STRIP_PREFIX_RE.sub("", q)[:100]
        records.append({
            "category": "replay_tool", "system": tool_system(rng, tools_block),
            "question": q, "answer": TOOL_CALL_TMPL.format(q=query_arg),
            "source": "synthetic_call", "seed": tmpl,
        })

    nocall_pool = reservoir_sample(BELLE_FILE, args.n_nocall, random.Random(args.seed + 1)) \
        if BELLE_FILE.exists() else []
    if len(nocall_pool) < args.n_nocall:
        print(f"警告: Belle 不可用或不足，no-call 仅 {len(nocall_pool)} 条（先跑 belle 子命令）")
    for r in nocall_pool:
        records.append({
            "category": "replay_tool", "system": tool_system(rng, tools_block),
            "question": r["q"], "answer": r["a"],
            "source": "belle_nocall", "seed": None,
        })

    rng.shuffle(records)
    out = RAW_DIR / "replay_tool.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[replay_tool] 写入 {len(records)} 条 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("belle")
    p1.add_argument("--n", type=int, default=800)
    p2 = sub.add_parser("tool")
    p2.add_argument("--n-call", type=int, default=90)
    p2.add_argument("--n-nocall", type=int, default=60)
    p2.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    (cmd_belle if args.cmd == "belle" else cmd_tool)(args)


if __name__ == "__main__":
    sys.exit(main())
