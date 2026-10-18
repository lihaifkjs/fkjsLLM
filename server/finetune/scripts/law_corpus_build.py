"""M12.3：把 `server/rag/corpus/` 的条文 txt 解析成结构化 index。

产出 `server/finetune/data/law/articles.json`：
  [{book, no_cn, no_int, label, text, n_paragraphs, n_chars}]
`label` 与 RAG chunk 的注入标签逐字一致（`《劳动合同法》第十九条`），
后续拼 SFT 样本的 context 直接用它，保证训练/推理同源。

用法：cd server && python finetune/scripts/law_corpus_build.py
"""
import json
import re
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2]
CORPUS_DIR = SERVER / "rag" / "corpus"
OUT = SERVER / "finetune" / "data" / "law" / "articles.json"
FILES = ["劳动合同法.txt", "劳动合同法实施条例.txt"]

_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")}
_DIGITS["〇"] = 0
_HEAD_RE = re.compile(r"^《([^》]+)》第([零〇一二三四五六七八九十百]+)条：(.*)$")


def cn2int(text: str) -> int:
    """中文数字 → int（支持到 百位，够 136 条用）。"""
    total = section = num = 0
    for ch in text:
        if ch in _DIGITS:
            num = _DIGITS[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
    return section + num


def main() -> None:
    articles = []
    for name in FILES:
        text = (CORPUS_DIR / name).read_text(encoding="utf-8")
        current = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            head = _HEAD_RE.match(line)
            if head:
                book, no_cn, body = head.groups()
                current = {
                    "book": book, "no_cn": no_cn, "no_int": cn2int(no_cn),
                    "label": f"《{book}》第{no_cn}条", "text": body,
                    "n_paragraphs": 1, "n_chars": len(body),
                }
                articles.append(current)
            elif current is not None:  # 续行：款/分项
                current["text"] += "\n" + line
                current["n_paragraphs"] += 1
                current["n_chars"] += len(line)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(articles, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    per_book = {}
    for a in articles:
        per_book[a["book"]] = per_book.get(a["book"], 0) + 1
    print(f"条文 {len(articles)} 条 → {OUT}")
    for book, n in per_book.items():
        print(f"  《{book}》{n} 条")
    # 连续性校验：条号必须从 1 连续到 N，且 label 无重复
    for book, n in per_book.items():
        nos = sorted(a["no_int"] for a in articles if a["book"] == book)
        assert nos == list(range(1, n + 1)), f"《{book}》条号不连续：{nos[:5]}…"
    assert len({a["label"] for a in articles}) == len(articles), "label 有重复"
    print("条号连续性与 label 唯一性校验通过")


if __name__ == "__main__":
    main()
