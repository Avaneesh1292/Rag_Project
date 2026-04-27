from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings() -> HuggingFaceEmbeddings:
    """Create and return the shared embedding model instance."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store(chunks, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Build a FAISS vector store from chunks using the provided embeddings."""
    return FAISS.from_documents(chunks, embeddings)
