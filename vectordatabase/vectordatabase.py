# %%
import sys
project_root = r"C:\Users\melis\OneDrive\Masaüstü\legal-research-assistant"
if project_root not in sys.path:
    sys.path.append(project_root)
    
from embeddings.embedding import embedding
from chunker.chunker import chunks
test_embedding = embedding.embed_query(
    "İşveren vekili nedir?"
)

print(len(test_embedding))

# %%
import os

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def create_vector_store(chunks, embedding):

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="is_kanunu"
    )

    return vectorstore

# %%



