import numpy as np
from typing import List, Dict, Any, Protocol
from rank_bm25 import BM25Okapi
from src.embedding import EmbeddingService, cosine_similarity
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from src.utils.logger import get_main_logger, get_rag_logger
from src.data_source import DataSource
import nltk
from src.utils.error_handler import handle_rag_error, RAGError
from src.config import EMBEDDING_DIMENSION, TOP_K_CHUNKS

# Initialize loggers for main and RAG-specific logging
logger = get_main_logger()
rag_logger = get_rag_logger()

# Define a protocol for search strategies
class SearchStrategy(Protocol):
    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        ...

# Base class for search strategies
class BaseSearch:
    def __init__(self, data_source: DataSource, embedding_service: EmbeddingService):
        self.data_source = data_source  # Data source containing chunks
        self.embedding_service = embedding_service  # Service for embedding operations
        self.chunks = data_source.get_chunks()  # Retrieve chunks from data source
        self.embeddings = data_source.get_embeddings()  # Retrieve embeddings from data source
        
        # Check if all embeddings have the correct dimension
        if any(len(emb) != EMBEDDING_DIMENSION for emb in self.embeddings):
            raise ValueError("Some embeddings have incorrect dimension")

    @handle_rag_error
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Abstract method to be implemented by subclasses
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        # Ensure scores is a numpy array
        if not isinstance(scores, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(scores)}")
        
        # Ensure scores contain numeric data
        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError(f"Array contains non-numeric data. scores dtype: {scores.dtype}")
        
        # Get indices of the top K scores
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': float(scores[i])} for i in top_indices]

# Hybrid search strategy combining BM25 and embedding-based search
class HybridSearch(BaseSearch):
    @handle_rag_error
    def __init__(self, data_source: DataSource):
        super().__init__(data_source)  # Initialize base class
        # Download necessary NLTK resources quietly
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Initialize BM25 with tokenized chunks
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Initialize lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()
        
        # Set embedding weight for combining scores
        self.embedding_weight = 0.5  # You can adjust this value as needed

    @handle_rag_error
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Expand the query using synonyms
            expanded_query = self._expand_query(query)
            # Get BM25 scores for the expanded query
            bm25_scores = self._get_bm25_scores(expanded_query)
            # Get embedding scores for the expanded query
            embedding_scores = self._get_embedding_scores(expanded_query)
            
            # Log shapes and types of scores for debugging
            logger.debug(f"BM25 scores shape: {bm25_scores.shape}, dtype: {bm25_scores.dtype}")
            logger.debug(f"Embedding scores shape: {embedding_scores.shape}, dtype: {embedding_scores.dtype}")
            
            # Combine BM25 and embedding scores
            combined_scores = self._combine_scores(bm25_scores, embedding_scores)
            
            # Log combined scores for debugging
            logger.debug(f"Combined scores shape: {combined_scores.shape}, dtype: {combined_scores.dtype}")
            
            # Return the top chunks based on combined scores
            return self._get_top_chunks(combined_scores, top_k)
        except Exception as e:
            raise RAGError(f"Error in hybrid search: {str(e)}")

    @handle_rag_error
    def _expand_query(self, query: str) -> str:
        # Tokenize the query and get part-of-speech tags
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)
        expanded_tokens = []

        # Expand tokens based on their POS tags
        for token, pos in pos_tags:
            expanded_tokens.append(token)  # Add original token
            if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                synsets = wordnet.synsets(token)  # Get synsets for the token
                for synset in synsets:
                    for lemma in synset.lemmas():
                        # Add lemma if it's not the original token and not already added
                        if lemma.name() != token and lemma.name() not in expanded_tokens:
                            expanded_tokens.append(lemma.name())

        return ' '.join(expanded_tokens)  # Return expanded query as a string

    @handle_rag_error
    def _get_bm25_scores(self, query: str) -> np.ndarray:
        # Get BM25 scores for the given query
        return np.array(self.bm25.get_scores(query.split()))

    @handle_rag_error
    def _get_embedding_scores(self, query: str) -> np.ndarray:
        # Create a weighted query embedding
        query_embedding = self._create_weighted_query_embedding(query)
        # Check the shape of the query embedding
        if isinstance(query_embedding, np.ndarray):
            query_shape = query_embedding.shape
        else:
            query_shape = (len(query_embedding),)
        # Check the shape of the chunk embeddings
        if isinstance(self.embeddings[0], np.ndarray):
            emb_shape = self.embeddings[0].shape
        else:
            emb_shape = (len(self.embeddings[0]),)
        
        # Ensure the shapes match
        if query_shape != emb_shape:
            raise ValueError(f"Query embedding shape {query_shape} does not match chunk embedding shape {emb_shape}")
        # Calculate cosine similarity scores for each chunk
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
        tokens = word_tokenize(query)  # Tokenize the query
        pos_tags = pos_tag(tokens)  # Get POS tags
        weighted_tokens = []

        # Assign weights based on POS tags
        for token, pos in pos_tags:
            weight = 1.0  # Default weight
            if pos.startswith('NN'):
                weight = 1.5  # Increase weight for nouns
            elif pos.startswith('VB'):
                weight = 1.3  # Increase weight for verbs
            elif pos.startswith('JJ'):
                weight = 1.2  # Increase weight for adjectives
        
            # Lemmatize the token and extend the weighted tokens list
            lemma = self.lemmatizer.lemmatize(token)
            weighted_tokens.extend([lemma] * int(weight * 10))  # Multiply by 10 to keep it as integer

        weighted_query = ' '.join(weighted_tokens)  # Join weighted tokens into a string
        return self.data_source.create_embedding(weighted_query)  # Create and return the embedding

    @handle_rag_error
    def _combine_scores(self, bm25_scores: np.ndarray, embedding_scores: np.ndarray) -> np.ndarray:
        # Ensure embedding_scores is not a string
        if isinstance(embedding_scores, str):
            raise ValueError(f"embedding_scores is a string: {embedding_scores}")
        # Normalize BM25 scores
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        # Normalize embedding scores
        embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        # Combine scores using the specified embedding weight
        return (1 - self.embedding_weight) * bm25_scores + self.embedding_weight * embedding_scores

    @handle_rag_error
    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        # Get indices of the top K scores
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': float(scores[i])} for i in top_indices]

class CosineSearch(BaseSearch):
    """Search implementation using cosine similarity between embeddings."""
    
    @handle_rag_error
    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks using cosine similarity between embeddings.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunks and their similarity scores
        """
        try:
            # Create embedding for the query
            query_embedding = self.embedding_service.create_embeddings([query])[0]
            # Calculate cosine similarity scores for each chunk
            scores = np.array([
                cosine_similarity(query_embedding, chunk_embedding) 
                for chunk_embedding in self.embeddings
            ])
            
            # Get the top chunks based on scores
            results = self._get_top_chunks(scores, top_k)
            
            # Log search results
            rag_logger.info(
                f"\nSearch Results:\n"
                f"Query: {query}\n"
                f"Top {top_k} results:\n" + 
                "\n".join([f"Score: {r['score']:.4f} | Chunk: {r['chunk'][:100]}..." 
                          for r in results]) +
                f"\n{'-'*50}"
            )
            
            return results  # Return the search results
        except Exception as e:
            # Log and raise an error if something goes wrong
            error_msg = f"Error in cosine similarity search: {str(e)}"
            logger.error(error_msg)
            rag_logger.error(f"\nSearch Error:\n{error_msg}\n{'-'*50}")
            raise RAGError(error_msg)

@handle_rag_error
def get_search_strategy(strategy: str, data_source: DataSource, embedding_service: EmbeddingService) -> BaseSearch:
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
        "cosine": CosineSearch,  # Cosine similarity search strategy
        "hybrid": HybridSearch,  # Hybrid search strategy
        # Planned strategies:
        # "semantic": SemanticSearch,
        # "fuzzy": FuzzySearch,
        # "contextual": ContextualSearch,
        # "index": IndexSearch,
        # "graph": GraphSearch,
    }
    
    # Fallback to cosine search if the strategy is unknown
    if strategy not in strategies:
        logger.warning(f"Unknown search strategy: {strategy}, using cosine similarity search")
        strategy = "cosine"
        
    return strategies[strategy](data_source, embedding_service)  # Передаем embedding_service
