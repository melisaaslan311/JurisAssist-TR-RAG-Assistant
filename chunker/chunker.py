import re
from langchain_core.documents import Document
import os
from pathlib import Path

def chunk_documents(documents):
    # sayfa metinlerini birleştir
    full_text = ""
    for document in documents:
        page_text = document.page_content.replace("\n", " ")
        page_text = re.sub(r"\s+", " ", page_text).strip()
        full_text += " " + page_text

    # madde başlangıçlarını yakalama
    pattern = re.compile(
        r"((?:Ek\s+|Geçici\s+)?Madde\s+\d+\s*[-–—])",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(full_text))
    raw_articles = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        article_body = full_text[start:end].strip()
        raw_articles.append({
            "match_text": match.group(1).rstrip(" -–—").strip(),
            "content": article_body
        })

    chunks = []
    next_title = ""

    for i in range(len(raw_articles)):
        content = raw_articles[i]["content"]
        article_num = raw_articles[i]["match_text"]

        if next_title:
            content = f"[{next_title}]\n" + content
            next_title = ""

        if i + 1 < len(raw_articles):
            tail_match = re.search(r'([.!?])\s+([A-ZÇĞİÖŞÜ0-9][^.!?]{1,60})$', content)
            if tail_match:
                extracted_title = tail_match.group(2).strip()
                content = content[:tail_match.start(2)].strip()
                next_title = extracted_title

        metadata = {
            "source": documents[0].metadata.get("source", "unknown") if documents else "unknown",
            "document_type": documents[0].metadata.get("document_type", "law") if documents else "law",
            "law_name": documents[0].metadata.get("law_name", "İş Kanunu") if documents else "unknown",
            "law_number": documents[0].metadata.get("law_number", "4857") if documents else "unknown",
            "article": article_num
        }

        chunks.append(Document(page_content=content, metadata=metadata))

    return chunks

BASE_DIR = Path(__file__).resolve().parent.parent
pdf_path = os.path.join(
    BASE_DIR, "data", "raw", "laws", "is_kanunu.pdf"
)

if __name__ == "__main__":
  from pdf_loader.pdf_loader import load_pdf
  documents = load_pdf(pdf_path)
  chunks = chunk_documents(documents)
  print(f"{len(chunks)} madde chunk'landı.")