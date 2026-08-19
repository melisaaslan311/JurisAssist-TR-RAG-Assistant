from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(pdf):
    path= Path(pdf)
    loader= PyPDFLoader(str(path))
    documents= loader.load()
    
    for document in documents:
        document.metadata.update({
            "source": path.name,
            "document_type": "law",
            "law_name": "is kanunu",
            "law_number": "4857"
        })
    
    return documents

BASE_DIR = Path(__file__).resolve().parent.parent
pdf_path = os.path.join(BASE_DIR, "data", "raw", "laws", "is_kanunu.pdf")

documents = load_pdf(pdf_path)
print(len(documents))




