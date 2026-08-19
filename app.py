import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("is_kanunu_rag")


@dataclass(frozen=True)
class AppConfig:
    qdrant_url: str
    qdrant_api_key: str
    google_api_key: str
    collection_name: str = "is_kanunu"
    gemini_model: str = "gemini-3.6-flash"
    retriever_k: int = 3
    page_title: str = "Hukuk Asistanı"
    page_icon: str = "⚖️"


def load_config() -> AppConfig:
    required = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise EnvironmentError(
            "Eksik ortam değişkenleri: " + ", ".join(missing) +
            ". Lütfen .env dosyanızı veya Render ortam değişkenlerini kontrol edin."
        )
    return AppConfig(
        qdrant_url=required["QDRANT_URL"],
        qdrant_api_key=required["QDRANT_API_KEY"],
        google_api_key=required["GOOGLE_API_KEY"],
        collection_name=os.getenv("QDRANT_COLLECTION", "is_kanunu"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        retriever_k=int(os.getenv("RETRIEVER_K", "3")),
    )


PROMPT_TEMPLATE = """Sen uzman bir Türk Hukuku ve İş Kanunu asistanısın.
Aşağıda sana verilen kanun maddelerini (BAĞLAM) kullanarak kullanıcının sorusunu yanıtla.

Kurallar:
1. Yalnızca verilen BAĞLAM içerisindeki bilgilere dayanarak cevap ver.
2. Tahminde bulunma; eğer cevap metinde yoksa "Verilen kanun metninde bu konuya dair bilgi bulunmamaktadır" de.
3. Cevabının sonunda mutlaka yararlandığın ilgili madde numaralarını (Kaynak) belirt.
4. Yanıtını açık, düzenli ve profesyonel bir dille, gerektiğinde maddeler halinde yaz.
5. Cevabına asla "BAĞLAM", "verilen bağlam", "verilen metin(ler)", "kanun metinleri çerçevesinde" gibisana nasıl bilgi verildiğine dair meta ifadelerle başlama veya bu ifadeleri kullanma. 
Doğrudan sorunun cevabıyla başla; sanki konuyu zaten biliyormuşsun gibi doğal bir hukuki dille yaz.


BAĞLAM:
{context}

SORU:
{question}

HUKUKİ CEVAP:"""


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "İlgili herhangi bir kanun maddesi bulunamadı."
    separator = "\n\n" + ("=" * 40) + "\n\n"
    formatted = []
    for doc in docs:
        article = doc.metadata.get("article", "Belirtilmemiş Madde")
        law_name = doc.metadata.get("law_name", "4857 Sayılı İş Kanunu")
        formatted.append(f"[{law_name} - {article}]\n{doc.page_content}")
    return separator.join(formatted)

config: AppConfig | None = None
rag_chain = None
retriever = None
init_error: str | None = None


def init_rag() -> None:
    global config, rag_chain, retriever, init_error
    try:
        config = load_config()

        from embeddings.embedding import embedding  # noqa: PLC0415 (projenize özel embedding modülü)

        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            collection_name=config.collection_name,
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k},
        )

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        llm = ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.google_api_key,
            temperature=0,
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG zinciri başarıyla kuruldu.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG kurulumunda hata oluştu.")
        init_error = str(exc)


init_rag()

app = Flask(__name__)


@app.route("/")
def index():
    cfg = config or AppConfig(qdrant_url="", qdrant_api_key="", google_api_key="")
    return render_template("index.html", page_title=cfg.page_title, page_icon=cfg.page_icon, init_error=init_error)


@app.route("/api/chat", methods=["POST"])
def chat():
    if init_error:
        return jsonify({"error": init_error}), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Soru boş olamaz."}), 400

    try:
        answer = rag_chain.invoke(question)
    except Exception:  # noqa: BLE001
        logger.exception("Yanıt üretilirken hata oluştu.")
        return jsonify({"error": "Üzgünüm, yanıt üretilirken bir hata oluştu. Lütfen tekrar deneyin."}), 500

    try:
        docs = retriever.invoke(question)
    except Exception:  # noqa: BLE001
        logger.exception("Kaynak belgeler getirilirken hata oluştu.")
        docs = []

    sources = [
        {
            "law_name": doc.metadata.get("law_name", "4857 Sayılı İş Kanunu"),
            "article": doc.metadata.get("article", "Madde"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    return jsonify({"answer": answer, "sources": sources})


@app.route("/health")
def health():
    status = "ok" if not init_error else "error"
    return jsonify({"status": status, "detail": init_error}), (200 if status == "ok" else 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
