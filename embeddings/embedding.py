import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embedding = get_embedding()
