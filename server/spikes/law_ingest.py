"""M12.1：把法律语料以「按条切块」模式灌入知识库（幂等，可重跑）。

走的是与上传接口同一条编排（rag.ingest.ingest_document），只是显式指定
chunk_mode="article"，所以网页端知识库列表能看到、能删，无需前端改动。

幂等：同名文档先删再灌（重新标定切块参数后直接重跑）。

用法：cd server && python spikes/law_ingest.py
"""
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIKE_DIR.parent))

from rag.ingest import ingest_document  # noqa: E402
from session_store.sqlite_store import session_store  # noqa: E402

CORPUS_DIR = SPIKE_DIR.parent / "rag" / "corpus"
CORPUS_FILES = ["劳动合同法.txt", "劳动合同法实施条例.txt"]


def main() -> None:
    for name in CORPUS_FILES:
        for doc in session_store.list_documents():
            if doc.filename == name:
                session_store.delete_document(doc.id)
                print(f"删除同名旧文档 #{doc.id} {name}")
        data = (CORPUS_DIR / name).read_bytes()
        doc = ingest_document(name, data, chunk_mode="article")
        print(f"入库 #{doc.id} {doc.filename}：{doc.char_count} 字 / "
              f"{doc.chunk_count} 块")


if __name__ == "__main__":
    main()
