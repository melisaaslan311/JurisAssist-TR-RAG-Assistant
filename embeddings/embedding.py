from langchain_huggingface import HuggingFaceEmbeddings

embedding= HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    encode_kwargs={
        "normalize_embeddings": True
    }
)