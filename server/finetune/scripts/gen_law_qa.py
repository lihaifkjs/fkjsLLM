"""M12.3：劳动法 SFT 语料的「骨架 → 生成 → 校验 → 落盘」流水线。

**为什么分「骨架」和「生成」两层**：context（用哪几条法条、什么形态）、题面、
gold 全部由 `scenarios.json` 静态定义，不依赖模型——这部分可人工精雕；模型只负责
把骨架写成带 think 的自然答案。这样模型的产出永远受骨架约束，不会自由发挥到
语料契约之外（M12 §5.1 / §8.4）。

四种 context 形态（对应「训练数据与 RAG 内容解耦」）：
  原文：直接用 articles.json 里的真实条文
  改写：scenarios 里给 `context_text`（同义改写，逼模型读上下文而不是背字串）
  要点：scenarios 里给 `points`（只给条号与要点，考补全推理）
  虚构：引用 virtual_articles.json 里编造的条文（条号不存在，机制上防记忆）

校验（不通过就重生成，默认重试 2 次，仍不过记入 rejects.jsonl）：
  - 必须含非空 `<think>`
  - 每个 `[n]` 的 n 在 context 范围内
  - `[n]` 前必须紧跟与材料一致的条文标签（契约：行内带出处）
  - insight 里不得出现 context 之外的条号（材料外幻觉）

用法:
  cd server
  # 只看骨架拼出来的 prompt（不加载模型，秒级）
  python finetune/scripts/gen_law_qa.py --dry-run
  # GPU 生成（.venv 里跑；-m 跳过已完成，可断点续跑）
  finetune/.venv/Scripts/python.exe finetune/scripts/gen_law_qa.py --backend hf
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER = SCRIPT_DIR.parents[1]
DATA_DIR = SERVER / "finetune" / "data" / "law"
ARTICLES = DATA_DIR / "articles.json"
VIRTUAL = DATA_DIR / "virtual_articles.json"
SCENARIOS = DATA_DIR / "scenarios.json"
OUT = DATA_DIR / "generated.jsonl"
REJECTS = DATA_DIR / "rejects.jsonl"
LOG = SERVER / "finetune" / "gen_law_qa.log"

ROLE = "你是一位中国大陆的劳动法咨询助手。"
RULES = (
    "规则：\n"
    "1. 只能使用【材料】里的内容作答，不得凭记忆补充条文；材料没有的，明说材料不足。\n"
    "2. 引用条文时写成「《法律名》第X条[n]」，n 是材料里的编号；条号必须与材料写法"
    "完全相同（中文数字，如「第一百二十六条」），不得改写成阿拉伯数字；"
    "不得只写 [n]，也不得引用材料外的条号。\n"
    "3. 先在 <think> 里判断，**不超过 80 字**；正文**不超过 250 字**，结构：结论 → "
    "依据 → 适用条件 → 可主张措施 → 一句免责。\n"
    "4. 拒答类（超出材料/超出范围）也给一段极短判定 think，然后简短拒答并说明下一步。\n"
    "5. 简短优先：宁可少写，也不要展开无关分支。"  # base 话多，长答案会顶到生成上限被截断
)

_CITE_RE = re.compile(r"\[(\d+)\]")
_LABEL_RE = re.compile(r"《[^》]{1,40}》第[零〇一二三四五六七八九十百]+条")
_ARTICLE_REF_RE = re.compile(r"第[零〇一二三四五六七八九十百]+条")
_ARABIC_ART_RE = re.compile(r"(?<=第)(\d{1,3})(?=条)")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# 判断标签是否紧邻 [n] 时要忽略的收尾标点/括号
_TRAILING = "）)】」』》”\"' \t"

_CN_DIGITS = "零一二三四五六七八九"


def cn_num(n: int) -> str:
    """1~999 → 中文数字（与法条写法一致：一百二十六 / 二十 / 三）。"""
    if n <= 0:
        return _CN_DIGITS[0]
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        return "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    if n < 100:
        return _CN_DIGITS[n // 10] + "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    head = _CN_DIGITS[n // 100] + "百"
    rest = n % 100
    if rest == 0:
        return head
    if rest < 10:
        return head + "零" + _CN_DIGITS[rest]
    return head + cn_num(rest)


# ---------------------------------------------------------------- 骨架 → prompt
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_material(context: list[dict]) -> str:
    """材料段与线上注入格式逐字一致：`[n] 《法》第X条：原文`。"""
    return "\n".join(f"[{c['ref']}] {c['title']}：{c['text']}" for c in context)


def build_context(sc: dict, articles: dict, virtual: dict) -> list[dict]:
    form = sc.get("form", "原文")
    if sc["category"] in ("law_insufficient", "law_scope_refuse") and not sc.get("context_refs"):
        return []
    if form == "虚构":
        return [{"ref": i + 1, "title": virtual[r]["label"], "text": virtual[r]["text"]}
                for i, r in enumerate(sc.get("virtual_refs", []))]
    if form == "改写":
        return [{"ref": i + 1, "title": t["title"], "text": t["text"]}
                for i, t in enumerate(sc.get("rewritten", []))]
    if form == "要点":
        return [{"ref": i + 1, "title": t["title"], "text": t["text"]}
                for i, t in enumerate(sc.get("points", []))]
    return [{"ref": i + 1, "title": articles[r]["label"], "text": articles[r]["text"]}
            for i, r in enumerate(sc.get("context_refs", []))]


FEWSHOT = [
    {  # 开卷作答范本
        "category": "law_grounded_qa",
        "question": "我签的是三年期合同，公司说试用期八个月，这样合法吗？",
        "context": [{"ref": 1, "title": "《劳动合同法》第十九条",
                     "text": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月；"
                             "一年以上不满三年的，不得超过二个月；三年以上固定期限和"
                             "无固定期限的，不得超过六个月。同一用人单位与同一劳动者"
                             "只能约定一次试用期。"}],
        "answer": "<think>\n三年以上固定期限，试用期上限六个月；八个月超限。\n</think>\n\n"
                  "不完全合法。三年期劳动合同的试用期不得超过六个月"
                  "（《劳动合同法》第十九条[1]），约定八个月超出法定上限。\n\n"
                  "适用条件：合同期限三年以上；同一单位与同一劳动者只能约定一次试用期"
                  "（《劳动合同法》第十九条[1]）。\n\n"
                  "可主张：要求改正试用期约定；已履行的超期部分可按试用期满月工资主张赔偿。\n\n"
                  "以上基于你提供的材料，具体金额与程序请结合当地规定，必要时咨询执业律师。",
    },
    {  # 材料不足范本
        "category": "law_insufficient",
        "question": "上海 2026 年的最低工资标准是多少钱一个月？",
        "context": [],
        "answer": "<think>\n材料为空。最低工资数额属地方标准，材料里没有，不能凭记忆给。\n</think>\n\n"
                  "未检索到可用资料，我不能凭记忆给出数字。\n\n"
                  "本问题需要「当地人社部门公布的最低工资标准」，当前材料未收录。\n\n"
                  "建议：补充当地最新公布的《最低工资标准》文件，或直接向当地人社部门查询。",
    },
    {  # 用户引错条号范本：材料里没有用户说的那条 → 明确指出，只用材料内的条作答
        "category": "law_grounded_qa",
        "question": "我记得第八十四条规定了试用期最长不得超过六个月，按那条给我看。",
        "context": [{"ref": 1, "title": "《劳动合同法》第十九条",
                     "text": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月；"
                             "一年以上不满三年的，不得超过二个月；三年以上固定期限和"
                             "无固定期限的，不得超过六个月。"}],
        "answer": "<think>\n材料只有第十九条，没有用户说的第八十四条。不能顺着错条号走。\n</think>\n\n"
                  "你提到的第八十四条不在我手上的材料里，我不能凭记忆引用它。\n\n"
                  "就材料内容而言：三年以上固定期限和无固定期限的劳动合同，试用期不得超过"
                  "六个月（《劳动合同法》第十九条[1]）。\n\n"
                  "建议核对条号出处后再作判断；如果方便，把该条文原文提供给我，我可以据此分析。",
    },
]


def build_messages(sc: dict, context: list[dict], turn: str | None = None,
                   history: list[dict] | None = None) -> list[dict]:
    """system（角色+规则）→ few-shot → 历史轮 → 当前轮。

    历史必须排在当前问题之前——否则模型会把上一轮重答一遍（冒烟实测踩过）。
    """
    messages = [{"role": "system", "content": ROLE + "\n\n" + RULES}]
    for ex in FEWSHOT:
        material = render_material(ex["context"]) or "（无）"
        messages.append({"role": "user",
                         "content": f"【材料】\n{material}\n\n【问题】\n{ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.extend(history or [])
    material = render_material(context) or "（无）"
    question = turn if turn is not None else (
        sc.get("turns") or [sc.get("question", "")])[-1]
    messages.append({"role": "user",
                     "content": f"【材料】\n{material}\n\n【问题】\n{question}"})
    return messages


# ---------------------------------------------------------------- 校验
def normalize_answer(answer: str, context: list[dict]) -> str:
    """把阿拉伯数字条号还原成中文写法（仅当材料里存在对应标签）。

    模型常把「第一百二十六条」写成「第126条」。这不是内容错误，但会让训练样本
    与注入材料的标签不一致（契约要求逐字一致），所以在校验前做无损规范化：
    只在能对上材料里的同号标签时替换，绝不改动条号数字本身。
    """
    labels = [c["title"] for c in context]

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        cn = cn_num(n)
        if any(label.endswith(f"第{cn}条") for label in labels):
            return cn
        return m.group(1)

    return _ARABIC_ART_RE.sub(repl, answer)


def validate(answer: str, context: list[dict], question: str = "") -> str | None:
    """返回 None=通过，否则返回拒绝原因。

    question 里出现的条号算「用户提到的」：诱导题要求模型明确指出「你说的第X条不在材料里」，
    这个 X 必须允许出现，否则正确答案会被判成材料外幻觉。
    """
    if "<think>" in answer and "</think>" not in answer:
        return "unterminated_think"  # 被生成上限截断，think 没闭合
    if not _THINK_RE.search(answer):
        return "no_think"
    labels = [c["title"] for c in context]
    body = _THINK_RE.sub("", answer)
    for m in _CITE_RE.finditer(body):
        n = int(m.group(1))
        if not 1 <= n <= len(labels):
            return f"cite_out_of_range:{n}"
        before = body[max(0, m.start() - 40):m.start()].rstrip().rstrip(_TRAILING)
        after = body[m.end():m.end() + 40].lstrip().lstrip(_TRAILING)
        head = list(_LABEL_RE.finditer(before))
        if head and before.endswith(head[-1].group(0)):
            written = head[-1].group(0)
        elif (tail := _LABEL_RE.match(after)):
            written = tail.group(0)  # 容忍「[1]《法》第X条」的写法
        else:
            return f"cite_without_label:{n}"
        if written != labels[n - 1]:
            return f"label_mismatch:{n}"
    labels = [c["title"] for c in context]
    # 「材料里出现过」的条号也算命中：法条正文常自带交叉引用（第八十七条原文就写着
    # 「按第四十七条标准」），把它当幻觉会误杀（冒烟实测 G005 连续两次被判违规）
    mentioned = {ref for c in context
                 for ref in _ARTICLE_REF_RE.findall(c["text"])}
    allowed = mentioned | set(_ARTICLE_REF_RE.findall(question or ""))
    outside = [r for r in _ARTICLE_REF_RE.findall(body)
               if r not in allowed and not any(r in label for label in labels)]
    if outside:
        return f"out_of_material:{outside[0]}"
    return None


# ---------------------------------------------------------------- 后端
class HfBackend:
    def __init__(self, model_dir: Path, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.torch, self.max_new_tokens = torch, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        self.model.eval()

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        texts = [self.tok.apply_chat_template(m, tokenize=False,
                                              add_generation_prompt=True)
                 for m in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        n_prompt = enc["input_ids"].shape[1]
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=True, temperature=0.7,
                                      top_p=0.9, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(seq, skip_special_tokens=True)
                for seq in out[:, n_prompt:]]


class LlamacppBackend:
    def __init__(self, model_path: Path, max_new_tokens: int):
        from llama_cpp import Llama, llama_supports_gpu_offload

        self.max_new_tokens = max_new_tokens
        self.llm = Llama(model_path=str(model_path), n_ctx=8192,
                         n_gpu_layers=-1 if llama_supports_gpu_offload() else 0,
                         verbose=False)

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        return [(self.llm.create_chat_completion(
            messages=m, temperature=0.7, top_p=0.9, max_tokens=self.max_new_tokens,
            stream=False)["choices"][0]["message"]["content"] or "")
            for m in prompts]


def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("hf", "llamacpp"), default="hf")
    ap.add_argument("--hf-model", default=str(SERVER / "finetune" / "models" / "Qwen3-14B"))
    ap.add_argument("--model", default=str(SERVER.parent / "Qwen3-14B-Q4_K_M.gguf"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只跑这些 id（逗号分隔），冒烟用")
    ap.add_argument("--dry-run", action="store_true", help="只拼 prompt 并打印，不加载模型")
    args = ap.parse_args()

    articles = {a["label"]: a for a in load_json(ARTICLES)}
    virtual = {v["id"]: v for v in load_json(VIRTUAL)}
    scenarios = load_json(SCENARIOS)["scenarios"]
    if args.limit:
        scenarios = scenarios[:args.limit]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["id"] in wanted]

    todo = []
    for sc in scenarios:
        context = build_context(sc, articles, virtual)
        todo.append({"sc": sc, "context": context,
                     "messages": build_messages(sc, context)})

    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOG.open("a", encoding="utf-8")

    def log(msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"骨架 {len(todo)} 条（{', '.join(sorted({t['sc']['category'] for t in todo}))}）")

    if args.dry_run:
        t = todo[0]
        log(f"示例 {t['sc']['id']}：{render_material(t['context'])[:120]}…")
        log(f"prompt 预览（末条 user）：\n{t['messages'][-1]['content'][:400]}")
        return 0

    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:  # 中断可能留下半截行：跳过无法解析的，其余照常续跑
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                log(f"!! 跳过损坏行：{line[:60]}")
    todo = [t for t in todo if t["sc"]["id"] not in done]
    log(f"已完成 {len(done)} 条，本次待生成 {len(todo)} 条")

    backend = HfBackend(Path(args.hf_model), args.max_new_tokens) \
        if args.backend == "hf" else LlamacppBackend(Path(args.model), args.max_new_tokens)

    # 单轮题整批解码（吞吐高）；多轮必须逐轮串行，前轮回答当历史
    single = [t for t in todo if len(t["sc"].get("turns") or []) <= 1]
    multi = [t for t in todo if len(t["sc"].get("turns") or []) > 1]

    t0 = time.perf_counter()
    ok = fail = 0

    def accept(item, answer, history=None):
        nonlocal ok
        sc = item["sc"]
        record = {
            "id": sc["id"], "category": sc["category"], "system": ROLE,
            "context": item["context"],
            "question": (sc.get("turns") or [sc.get("question")])[-1],
            "answer": answer.strip(),
        }
        if history:  # 多轮：history 含末轮 assistant，末轮正文单独放 answer
            record["messages"] = history[:-1]
        elif sc.get("turns"):
            record["messages"] = [{"role": "user", "content": t}
                                  for t in sc["turns"]]
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        ok += 1
        log(f"  ✓ {sc['id']} {len(answer)} 字")

    def give_up(item):
        nonlocal fail
        with REJECTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["sc"]["id"],
                                "category": item["sc"]["category"],
                                "reason": "retries_exhausted"},
                               ensure_ascii=False) + "\n")
        fail += 1

    for i in range(0, len(single), args.batch):
        chunk = single[i:i + args.batch]
        attempts = {id(t): t for t in chunk}
        for round_no in range(args.retries + 1):
            if not attempts:
                break
            keys = list(attempts)
            answers = backend.generate([attempts[k]["messages"] for k in keys])
            for key, answer in zip(keys, answers):
                item = attempts[key]
                fixed = normalize_answer(answer, item["context"])
                if fixed != answer:
                    log(f"  ~ {item['sc']['id']} 条号数字已规范化")
                reason = validate(fixed, item["context"],
                                  item["sc"].get("question", ""))
                if reason is None:
                    accept(item, fixed)
                    del attempts[key]
                else:
                    log(f"  ✗ {item['sc']['id']} 第{round_no + 1}次：{reason}")
                    item["messages"] = build_messages(item["sc"], item["context"])
        for item in attempts.values():
            give_up(item)
        log(f"进度 {min(i + args.batch, len(single))}/{len(single)}"
            f"（ok {ok} / fail {fail}，{hms(time.perf_counter() - t0)}）")

    for item in multi:  # 多轮：逐轮生成并累积历史，只对末轮做校验
        sc, context = item["sc"], item["context"]
        for round_no in range(args.retries + 1):
            history, answer = [], ""
            for turn in sc["turns"]:
                messages = build_messages(sc, context, turn, history)
                answer = backend.generate([messages])[0]
                history += [{"role": "user", "content": turn},
                            {"role": "assistant", "content": answer}]
            answer = normalize_answer(answer, context)
            reason = validate(answer, context, sc["turns"][-1])
            if reason is None:
                accept(item, answer, history)
                break
            log(f"  ✗ {sc['id']} 多轮第{round_no + 1}次：{reason}")
        else:
            give_up(item)

    log(f"完成：采纳 {ok} 条，弃用 {fail} 条 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
