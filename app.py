"""
İş Kanunu RAG Asistanı
=======================
4857 Sayılı Türk İş Kanunu üzerinde RAG (Retrieval-Augmented Generation)
tabanlı soru-cevap sağlayan Streamlit uygulaması.

Çalıştırmak için:
    streamlit run app.py

Gerekli ortam değişkenleri (.env):
    QDRANT_URL
    QDRANT_API_KEY
    GOOGLE_API_KEY
    GEMINI_MODEL (opsiyonel, varsayılan: gemini-3.6-flash)
"""
import os
# HuggingFace'in her seferinde internete bağlanıp versiyon kontrolü yapmasını engeller
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import html
import logging
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore


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
    """Ortam değişkenlerini yükler ve doğrular."""
    load_dotenv()

    required = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise EnvironmentError(
            "Eksik ortam değişkenleri: " + ", ".join(missing) +
            ". Lütfen .env dosyanızı kontrol edin."
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

BAĞLAM:
{context}

SORU:
{question}

HUKUKİ CEVAP:"""


def format_docs(docs: list[Document]) -> str:
    """Getirilen belgeleri modele verilecek bağlam metnine dönüştürür."""
    if not docs:
        return "İlgili herhangi bir kanun maddesi bulunamadı."

    separator = "\n\n" + ("=" * 40) + "\n\n"
    formatted = []
    for doc in docs:
        article = doc.metadata.get("article", "Belirtilmemiş Madde")
        law_name = doc.metadata.get("law_name", "4857 Sayılı İş Kanunu")
        formatted.append(f"[{law_name} - {article}]\n{doc.page_content}")
    return separator.join(formatted)


# ---------------------------------------------------------------------------
# Kaynakların önbelleğe alınması (vectorstore, retriever, chain)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Bilgi tabanına bağlanılıyor...")
def get_vectorstore(_config: AppConfig) -> QdrantVectorStore:
    from embeddings.embedding import embedding 
    try:
        return QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            collection_name=_config.collection_name,
            url=_config.qdrant_url,
            api_key=_config.qdrant_api_key,
        )
    except Exception as exc: 
        logger.exception("Qdrant koleksiyonuna bağlanırken hata oluştu.")
        raise RuntimeError(
            "Vektör veritabanına bağlanılamadı. Qdrant ayarlarınızı kontrol edin."
        ) from exc


@st.cache_resource(show_spinner="RAG zinciri hazırlanıyor...")
def get_rag_chain(_config: AppConfig):
    vectorstore = get_vectorstore(_config)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": _config.retriever_k},
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatGoogleGenerativeAI(
        model=_config.gemini_model,
        google_api_key=_config.google_api_key,
        temperature=0,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=Source+Serif+4:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
    --ink:#1B2A4A;
    --ink-light:#2E4270;
    --paper:#FAF6EC;
    --paper-card:#FFFDF6;
    --gold:#A9822F;
    --gold-soft:#D8C48F;
    --seal:#7A2E2E;
    --text:#2B2620;
    --text-muted:#6E6656;
}

/* --- Zemin --- */
.stApp{
    background:
        repeating-linear-gradient(180deg, rgba(169,130,47,0.035) 0px, rgba(169,130,47,0.035) 1px, transparent 1px, transparent 34px),
        var(--paper);
}
.stApp, .stMarkdown, p, span, label, .stCaption{
    font-family:'Source Serif 4', Georgia, serif;
    color:var(--text);
}
h1,h2,h3,h4{
    font-family:'Playfair Display', Georgia, serif !important;
    color:var(--ink) !important;
    letter-spacing:.2px;
}

/* --- Masthead (Resmî Gazete tarzı üst başlık) --- */
.ik-masthead{
    text-align:center;
    padding:8px 0 22px 0;
    margin-bottom:10px;
    border-bottom:3px double var(--gold);
}
.ik-eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    letter-spacing:3px;
    color:var(--seal);
    text-transform:uppercase;
    margin-bottom:6px;
}
.ik-title{
    font-family:'Playfair Display', Georgia, serif;
    font-weight:700;
    font-size:2.3rem;
    color:var(--ink);
    margin:0;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:12px;
}
.ik-seal{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:44px;height:44px;
    border-radius:50%;
    background:radial-gradient(circle at 35% 30%, #8f3939, var(--seal) 70%);
    color:var(--gold-soft);
    font-size:20px;
    box-shadow:0 0 0 2px var(--gold-soft), inset 0 0 6px rgba(0,0,0,.35);
}
.ik-subtitle{
    font-family:'Source Serif 4', Georgia, serif;
    font-style:italic;
    color:var(--text-muted);
    font-size:.98rem;
    margin-top:8px;
}

/* --- Sohbet baloncukları --- */
[data-testid="stChatMessage"]{
    background:var(--paper-card);
    border:1px solid var(--gold-soft);
    border-radius:4px;
    box-shadow:0 1px 3px rgba(43,38,32,.06);
    padding:2px 4px;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{
    background:var(--ink) !important;
}

/* --- Girdi kutusu --- */
[data-testid="stChatInput"]{
    border:1px solid var(--gold);
    border-radius:6px;
    background:var(--paper-card);
}
[data-testid="stChatInput"] textarea{
    font-family:'Source Serif 4', Georgia, serif !important;
    color:var(--text) !important;
}

/* --- Butonlar --- */
.stButton>button{
    font-family:'IBM Plex Mono', monospace;
    letter-spacing:.5px;
    background:var(--ink);
    color:var(--paper);
    border:1px solid var(--ink);
    border-radius:4px;
    transition:all .15s ease;
}
.stButton>button:hover{
    background:var(--seal);
    border-color:var(--seal);
    color:#fff;
}

/* --- Kenar çubuğu ("dosya" paneli) --- */
[data-testid="stSidebar"]{
    background:var(--ink);
    border-right:3px solid var(--gold);
}
[data-testid="stSidebar"] *{
    color:var(--paper) !important;
    font-family:'Source Serif 4', Georgia, serif;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{
    font-family:'Playfair Display', Georgia, serif !important;
    color:var(--gold-soft) !important;
}
[data-testid="stSidebar"] hr{ border-color:rgba(216,196,143,.35) !important; }
/* İkon fontlarını serif geçersiz kılmadan koru (ör. çökertme oku) */
[data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .material-symbols-rounded,
[data-testid="stSidebar"] [class*="material-symbols"]{
    font-family:'Material Symbols Rounded' !important;
    color:var(--gold-soft) !important;
}
[data-testid="collapsedControl"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]{
    font-family:'Material Symbols Rounded' !important;
}
[data-testid="stSidebar"] code{
    background:rgba(216,196,143,.12) !important;
    color:var(--gold-soft) !important;
}
[data-testid="stAlert"]{
    background:rgba(216,196,143,.1) !important;
    border:1px solid var(--gold-soft) !important;
    border-radius:4px;
}

/* --- Kaynak (madde) mühür rozetleri --- */
.ik-source-card{
    border:1px solid var(--gold-soft);
    background:var(--paper-card);
    border-radius:4px;
    padding:12px 14px;
    margin-bottom:10px;
}
.ik-badge{
    display:inline-flex;
    align-items:center;
    gap:7px;
    font-family:'IBM Plex Mono', monospace;
    font-size:12.5px;
    font-weight:600;
    letter-spacing:.4px;
    color:var(--seal);
    background:#fff;
    border:1px solid var(--seal);
    border-radius:999px;
    padding:3px 11px 3px 6px;
    margin-bottom:8px;
}
.ik-badge .dot{
    width:8px;height:8px;border-radius:50%;
    background:var(--seal);
    box-shadow:0 0 0 2px rgba(122,46,46,.18);
}
.ik-source-text{
    font-family:'Source Serif 4', Georgia, serif;
    font-size:.92rem;
    color:var(--text-muted);
    line-height:1.5;
}

/* --- Expander başlığı --- */
[data-testid="stExpander"]{
    border:1px solid var(--gold-soft) !important;
    border-radius:4px !important;
    background:var(--paper-card);
}
[data-testid="stExpander"] summary{
    font-family:'IBM Plex Mono', monospace !important;
    color:var(--ink) !important;
}
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_masthead(config: AppConfig) -> None:
    st.markdown(
        f"""
        <div class="ik-masthead">
            <div class="ik-eyebrow">4857 Sayılı Türk İş Kanunu &nbsp;·&nbsp; Kaynak Gösteren Yapay Zekâ</div>
            <div class="ik-title"><span class="ik-seal">{config.page_icon}</span> İş Kanunu RAG Asistanı</div>
            <div class="ik-subtitle">Sorunuzu sorun — cevap, ilgili kanun maddelerine dayandırılır ve kaynağıyla gösterilir.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: AppConfig) -> None:
    with st.sidebar:
        st.markdown("### ⚖️ Dosya Bilgisi")

        st.divider()
        if st.button("🗑️  Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.info(
            "Bu asistan yalnızca 4857 Sayılı İş Kanunu metnine dayanarak "
            "genel bilgilendirme amaçlı cevaplar üretir. Hukuki bağlayıcılığı "
            "yoktur; önemli kararlar için bir avukata danışınız.",
            icon="ℹ️",
        )


def render_sources(docs: list[Document]) -> None:
    with st.expander("📚  Kullanılan Kanun Maddeleri", expanded=False):
        if not docs:
            st.markdown(
                '<div class="ik-source-text">İlgili bir kaynak bulunamadı.</div>',
                unsafe_allow_html=True,
            )
            return
        for doc in docs:
            article = html.escape(str(doc.metadata.get("article", "Madde")))
            law_name = html.escape(str(doc.metadata.get("law_name", "4857 Sayılı İş Kanunu")))
            content = html.escape(doc.page_content)
            st.markdown(
                f"""
                <div class="ik-source-card">
                    <div class="ik-badge"><span class="dot"></span>{law_name} — {article}</div>
                    <div class="ik-source-text">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history() -> None:
    for message in st.session_state.messages:
        avatar = "⚖️" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def main() -> None:
    try:
        config = load_config()
    except EnvironmentError as exc:
        st.set_page_config(page_title="Hukuk Asistanı", page_icon="⚖️", layout="centered")
        inject_custom_css()
        st.error(str(exc))
        st.stop()
        return

    st.set_page_config(page_title=config.page_title, page_icon=config.page_icon, layout="centered")
    inject_custom_css()
    render_masthead(config)

    render_sidebar(config)

    try:
        rag_chain, retriever = get_rag_chain(config)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_history()

    user_question = st.chat_input("Sorunuzu buraya yazın...")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="⚖️"):
        try:
            response = st.write_stream(rag_chain.stream(user_question))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Yanıt üretilirken hata oluştu.")
            response = (
                "Üzgünüm, yanıt üretilirken bir hata oluştu. "
                "Lütfen daha sonra tekrar deneyin."
            )
            st.error(response)
        else:
            try:
                docs = retriever.invoke(user_question)
            except Exception:  # noqa: BLE001
                logger.exception("Kaynak belgeler getirilirken hata oluştu.")
                docs = []
            render_sources(docs)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()