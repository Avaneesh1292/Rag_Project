import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config import SEMANTIC_K, BM25_K, RERANK_K

# Load once at import time — shared across all HybridRetriever instances
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class HybridRetriever:
    def __init__(self, vector_store, embeddings, chunks):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunks = chunks
        self.bm25 = BM25Okapi(
            [c.page_content.lower().split() for c in chunks]
        )

    def retrieve(self, query: str):
        """Hybrid retrieval: semantic (FAISS) + keyword (BM25), deduplicated."""
        semantic = self.vector_store.similarity_search(query, k=SEMANTIC_K)

        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:BM25_K]
        bm_docs = [self.chunks[i] for i in top]

        # Deduplicate by content
        merged = {d.page_content: d for d in semantic + bm_docs}
        return list(merged.values())

    def rerank(self, query: str, docs: list):
        """
        Cross-encoder reranking.

        Scores each (query, doc) pair jointly using a dedicated cross-encoder
        model, which is significantly more accurate than cosine similarity
        applied to independent embeddings.
        """
        if not docs:
            return docs

        pairs = [[query, d.page_content] for d in docs]
        scores = _cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:RERANK_K]]
