import numpy as np
from typing import List, Dict, Any, Protocol
from rank_bm25 import BM25Okapi
from src.embedding import EmbeddingService, cosine_similarity
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from src.utils.logger import setup_logger
from src.data_source import DataSource
import nltk
from src.utils.error_handler import handle_rag_error, RAGError
from src.config import EMBEDDING_DIMENSION

logger = setup_logger()

class SearchStrategy(Protocol):
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...

class BaseSearch:
    def __init__(self, data_source: DataSource, embedding_service: EmbeddingService):
        self.data_source = data_source
        self.embedding_service = embedding_service
        self.chunks = data_source.get_chunks()
        self.embeddings = data_source.get_embeddings()
        
        # Проверка размерности эмбеддингов
        if any(len(emb) != EMBEDDING_DIMENSION for emb in self.embeddings):
            raise ValueError("Some embeddings have incorrect dimension")

    @handle_rag_error
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        # Проверка типа данных
        if not isinstance(scores, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(scores)}")
        
        # Проверка на наличие числовых данных
        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError(f"Array contains non-numeric data. scores dtype: {scores.dtype}")
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': float(scores[i])} for i in top_indices]


class HybridSearch(BaseSearch):
    @handle_rag_error
    def __init__(self, data_source: DataSource):
        super().__init__(data_source)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Initialize BM25
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Set embedding weight
        self.embedding_weight = 0.5  # You can adjust this value as needed

    @handle_rag_error
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            expanded_query = self._expand_query(query)
            bm25_scores = self._get_bm25_scores(expanded_query)
            embedding_scores = self._get_embedding_scores(expanded_query)
            
            logger.debug(f"BM25 scores shape: {bm25_scores.shape}, dtype: {bm25_scores.dtype}")
            logger.debug(f"Embedding scores shape: {embedding_scores.shape}, dtype: {embedding_scores.dtype}")
            
            combined_scores = self._combine_scores(bm25_scores, embedding_scores)
            
            logger.debug(f"Combined scores shape: {combined_scores.shape}, dtype: {combined_scores.dtype}")
            
            return self._get_top_chunks(combined_scores, top_k)
        except Exception as e:
            raise RAGError(f"Error in hybrid search: {str(e)}")

    @handle_rag_error
    def _expand_query(self, query: str) -> str:
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)
        expanded_tokens = []

        for token, pos in pos_tags:
            expanded_tokens.append(token)
            if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                synsets = wordnet.synsets(token)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        if lemma.name() != token and lemma.name() not in expanded_tokens:
                            expanded_tokens.append(lemma.name())

        return ' '.join(expanded_tokens)

    @handle_rag_error
    def _get_bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.split()))

    @handle_rag_error
    def _get_embedding_scores(self, query: str) -> np.ndarray:
        query_embedding = self._create_weighted_query_embedding(query)
        if isinstance(query_embedding, np.ndarray):
            query_shape = query_embedding.shape
        else:
            query_shape = (len(query_embedding),)
        if isinstance(self.embeddings[0], np.ndarray):
            emb_shape = self.embeddings[0].shape
        else:
            emb_shape = (len(self.embeddings[0]),)
        
        if query_shape != emb_shape:
            raise ValueError(f"Query embedding shape {query_shape} does not match chunk embedding shape {emb_shape}")
        return np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])

    @handle_rag_error
    def _create_weighted_query_embedding(self, query: str) -> List[float]:
        """
        Create weighted query embedding based on POS tags.
        
        Args:
            query: Search query
        
        Returns:
            Weighted query embedding
        """
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
        return self.data_source.create_embedding(weighted_query)

    @handle_rag_error
    def _combine_scores(self, bm25_scores: np.ndarray, embedding_scores: np.ndarray) -> np.ndarray:
        if isinstance(embedding_scores, str):
            raise ValueError(f"embedding_scores is a string: {embedding_scores}")
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        return (1 - self.embedding_weight) * bm25_scores + self.embedding_weight * embedding_scores

    @handle_rag_error
    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': float(scores[i])} for i in top_indices]

class CosineSimilaritySearch(BaseSearch):
    """Search implementation using cosine similarity between embeddings."""
    
    @handle_rag_error
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks using cosine similarity between embeddings.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunks and their similarity scores
        """
        try:
            query_embedding = self.data_source.create_embedding(query)
            scores = np.array([
                cosine_similarity(query_embedding, chunk_embedding) 
                for chunk_embedding in self.embeddings
            ])
            return self._get_top_chunks(scores, top_k)
        except Exception as e:
            raise RAGError(f"Error in cosine similarity search: {str(e)}")

class CosineSearch(BaseSearch):
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for most similar chunks using cosine similarity."""
        try:
            # Создаем эмбеддинг для запроса через сервис
            query_embedding = self.embedding_service.create_embedding(query)
            
            # Вычисляем косинусное сходство
            scores = np.array([
                cosine_similarity(query_embedding, chunk_embedding) 
                for chunk_embedding in self.embeddings
            ])
            
            return self._get_top_chunks(scores, top_k)
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {str(e)}")
            raise RAGError(f"Error in cosine similarity search: {str(e)}")

@handle_rag_error
def get_search_strategy(strategy: str, data_source: DataSource) -> BaseSearch:
    """
    Get search strategy based on name.
    
    Available strategies:
    - cosine: Simple cosine similarity search using embeddings
    - hybrid: Combined BM25 and embedding-based search with query expansion
    - semantic: (planned) Pure semantic search with advanced NLP
    - fuzzy: (planned) Fuzzy string matching for typo tolerance
    - contextual: (planned) Context-aware search using document structure
    
    Args:
        strategy: Name of the search strategy
        data_source: Data source containing chunks and embeddings
        
    Returns:
        Initialized search strategy
    """
    strategies = {
        "cosine": CosineSimilaritySearch,
        "hybrid": HybridSearch,
        # Планируемые стратегии:
        # "semantic": SemanticSearch,
        # "fuzzy": FuzzySearch,
        # "contextual": ContextualSearch,
        # "index": IndexSearch,
        # "graph": GraphSearch,
    }
    
    if strategy not in strategies:
        logger.warning(f"Unknown search strategy: {strategy}, using cosine similarity search")
        strategy = "cosine"
        
    return strategies[strategy](data_source)
