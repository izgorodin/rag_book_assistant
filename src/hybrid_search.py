import numpy as np
from typing import List, Dict, Any, Union
from rank_bm25 import BM25Okapi
from src.embedding import create_embeddings, cosine_similarity
import logging

logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self, chunks: List[str], embeddings: List[List[float]], embedding_weight: float = 0.75):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_weight = embedding_weight
        self.bm25 = self._initialize_bm25()

    def _initialize_bm25(self):
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        return BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        try:
            bm25_scores = self._get_bm25_scores(query)
            embedding_scores = self._get_embedding_scores(query)
            combined_scores = self._combine_scores(bm25_scores, embedding_scores)
            return self._get_top_chunks(combined_scores, top_k)
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.split()))

    def _get_embedding_scores(self, query: str) -> np.ndarray:
        query_embedding = create_embeddings([query])[0]
        return np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])

    def _combine_scores(self, bm25_scores: np.ndarray, embedding_scores: np.ndarray) -> np.ndarray:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        return (1 - self.embedding_weight) * bm25_scores + self.embedding_weight * embedding_scores

    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': scores[i]} for i in top_indices]
