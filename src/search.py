import numpy as np
from typing import List, Dict, Any, Protocol, Tuple
from rank_bm25 import BM25Okapi
from src.book_data_interface import BookDataInterface
from src.embedding import create_embeddings, cosine_similarity
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from src.logger import setup_logger
from src.data_source import DataSource
import nltk
from src.error_handler import handle_rag_error, RAGError, SimpleSearchError, HybridSearchError, QueryExpansionError, ScoreComputationError, DimensionMismatchError
from src.config import TEXT_PROCESSING_CONFIG
from src.types import (
    Chunk, EmbeddingList, QueryType, TopK, SearchResults, ExpandedQuery, Weight,
    TokenizedChunks, Embedding
)

logger = setup_logger()

class SearchStrategy(Protocol):
    def search(self, query: QueryType, top_k: TopK = TopK(5)) -> SearchResults:
        ...

class BaseSearch:
    @handle_rag_error
    def __init__(self, data_source: DataSource):
        self.data_source: DataSource = data_source
        self.chunks: List[Chunk] = data_source.get_chunks()
        self.embeddings: EmbeddingList = data_source.get_embeddings()
        
        if any(len(emb) != TEXT_PROCESSING_CONFIG['dimension'] for emb in self.embeddings):
            raise ValueError("Some embeddings have incorrect dimension")

    @handle_rag_error
    def search(self, query: QueryType, top_k: TopK = TopK(5)) -> SearchResults:
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_top_chunks(self, scores: np.ndarray, top_k: TopK) -> SearchResults:
        if not isinstance(scores, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(scores)}")
        
        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError(f"Array contains non-numeric data. scores dtype: {scores.dtype}")
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': float(scores[i])} for i in top_indices]

class SimpleSearch(BaseSearch):
    @handle_rag_error
    def __init__(self, data_source: BookDataInterface):
        self.chunks = data_source.get_chunks()
        self.embeddings = data_source.get_embeddings()

    @handle_rag_error
    def search(self, query: QueryType, top_k: TopK = TopK(5)) -> SearchResults:
        try:
            query_embedding: Embedding = create_embeddings([query])[0]
            if len(query_embedding) != TEXT_PROCESSING_CONFIG['dimension']:
                raise ValueError(f"Query embedding has incorrect dimension: {len(query_embedding)}")
            scores: np.ndarray = np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])
            return self._get_top_chunks(scores, top_k)
        except Exception as e:
            raise SimpleSearchError(f"Error in simple search: {str(e)}")

class HybridSearch(BaseSearch):
    @handle_rag_error
    def __init__(self, data_source: DataSource):
        super().__init__(data_source)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        
        tokenized_chunks: TokenizedChunks = [chunk.split() for chunk in self.chunks]
        self.bm25: BM25Okapi = BM25Okapi(tokenized_chunks)
        
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        
        self.embedding_weight: Weight = Weight(0.5)  # You can adjust this value as needed

    @handle_rag_error
    def search(self, query: QueryType, top_k: TopK = TopK(5)) -> SearchResults:
        try:
            expanded_query: ExpandedQuery = self._expand_query(query)
            bm25_scores: np.ndarray = self._get_bm25_scores(expanded_query)
            embedding_scores: np.ndarray = self._get_embedding_scores(expanded_query)
            
            logger.debug(f"BM25 scores shape: {bm25_scores.shape}, dtype: {bm25_scores.dtype}")
            logger.debug(f"Embedding scores shape: {embedding_scores.shape}, dtype: {embedding_scores.dtype}")
            
            combined_scores: np.ndarray = self._combine_scores(bm25_scores, embedding_scores)
            
            logger.debug(f"Combined scores shape: {combined_scores.shape}, dtype: {combined_scores.dtype}")
            
            return self._get_top_chunks(combined_scores, top_k)
        except Exception as e:
            raise HybridSearchError(f"Error in hybrid search: {str(e)}")

    @handle_rag_error
    def _expand_query(self, query: QueryType) -> ExpandedQuery:
        try:
            tokens: List[str] = nltk.word_tokenize(query)
            pos_tags: List[Tuple[str, str]] = nltk.pos_tag(tokens)
            expanded_tokens: List[str] = []

            for token, pos in pos_tags:
                expanded_tokens.append(token)
                if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                    synsets = wordnet.synsets(token)
                    for synset in synsets:
                        for lemma in synset.lemmas():
                            if lemma.name() != token and lemma.name() not in expanded_tokens:
                                expanded_tokens.append(lemma.name())

            return ExpandedQuery(' '.join(expanded_tokens))
        except Exception as e:
            raise QueryExpansionError(f"Error in query expansion: {str(e)}")

    @handle_rag_error
    def _get_bm25_scores(self, query: ExpandedQuery) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.split()))

    @handle_rag_error
    def _get_embedding_scores(self, query: ExpandedQuery) -> np.ndarray:
        query_embedding: Embedding = self._create_weighted_query_embedding(query)
        if query_embedding.shape != self.embeddings[0].shape:
            raise ValueError(f"Query embedding shape {query_embedding.shape} does not match chunk embedding shape {self.embeddings[0].shape}")
        return np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])

    @handle_rag_error
    def _create_weighted_query_embedding(self, query: ExpandedQuery) -> Embedding:
        tokens: List[str] = word_tokenize(query)
        pos_tags: List[Tuple[str, str]] = pos_tag(tokens)
        weighted_tokens: List[str] = []

        for token, pos in pos_tags:
            weight: float = 1.0
            if pos.startswith('NN'):
                weight = 1.5  # Increase weight for nouns
            elif pos.startswith('VB'):
                weight = 1.3  # Increase weight for verbs
            elif pos.startswith('JJ'):
                weight = 1.2  # Increase weight for adjectives
            
            lemma: str = self.lemmatizer.lemmatize(token)
            weighted_tokens.extend([lemma] * int(weight * 10))  # Multiply by 10 to keep it as integer

        weighted_query: str = ' '.join(weighted_tokens)
        return Embedding(create_embeddings([weighted_query])[0])

    @handle_rag_error
    def _combine_scores(self, bm25_scores: np.ndarray, embedding_scores: np.ndarray) -> np.ndarray:
        if isinstance(embedding_scores, str):
            raise ValueError(f"embedding_scores is a string: {embedding_scores}")
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        return (1 - self.embedding_weight) * bm25_scores + self.embedding_weight * embedding_scores

@handle_rag_error
def get_search_strategy(strategy: str, data_source: DataSource) -> BaseSearch:
    if strategy == "simple":
        return SimpleSearch(data_source)
    elif strategy == "hybrid":
        return HybridSearch(data_source)
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")
