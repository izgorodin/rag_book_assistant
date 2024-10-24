import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.embedding import create_embeddings, cosine_similarity
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from src.logger import setup_logger

logger = setup_logger()

class HybridSearch:
    def __init__(self, chunks: List[str], embeddings: List[List[float]], embedding_weight: float = 0.75):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_weight = embedding_weight
        self.bm25 = self._initialize_bm25()
        self.lemmatizer = WordNetLemmatizer()

    def _initialize_bm25(self):
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        return BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        try:
            expanded_query = self._expand_query(query)
            bm25_scores = self._get_bm25_scores(expanded_query)
            embedding_scores = self._get_embedding_scores(expanded_query)
            combined_scores = self._combine_scores(bm25_scores, embedding_scores)
            return self._get_top_chunks(combined_scores, top_k)
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def _expand_query(self, query: str) -> str:
        tokens = word_tokenize(query)
        pos_tags = pos_tag(tokens)
        expanded_tokens = []

        for token, pos in pos_tags:
            expanded_tokens.append(token)
            if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                synonyms = self._get_synonyms(token)
                expanded_tokens.extend(synonyms[:2])  # Add up to 2 synonyms

        return ' '.join(expanded_tokens)

    def _get_synonyms(self, word: str) -> List[str]:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.split()))

    def _get_embedding_scores(self, query: str) -> np.ndarray:
        query_embedding = self._create_weighted_query_embedding(query)
        return np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])

    def _create_weighted_query_embedding(self, query: str) -> List[float]:
        tokens = word_tokenize(query)
        pos_tags = pos_tag(tokens)
        weighted_tokens = []

        for token, pos in pos_tags:
            weight = 1.0
            if pos.startswith('NN'):
                weight = 1.5  # Increase weight for nouns
            elif pos.startswith('VB'):
                weight = 1.3  # Increase weight for verbs
            elif pos.startswith('JJ'):
                weight = 1.2  # Increase weight for adjectives
            
            lemma = self.lemmatizer.lemmatize(token)
            weighted_tokens.extend([lemma] * int(weight * 10))  # Multiply by 10 to keep it as integer

        weighted_query = ' '.join(weighted_tokens)
        return create_embeddings([weighted_query])[0]

    def _combine_scores(self, bm25_scores: np.ndarray, embedding_scores: np.ndarray) -> np.ndarray:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        return (1 - self.embedding_weight) * bm25_scores + self.embedding_weight * embedding_scores

    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': scores[i]} for i in top_indices]
