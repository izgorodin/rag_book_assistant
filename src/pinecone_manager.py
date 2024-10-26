from abc import abstractmethod, ABC
from contextlib import contextmanager
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Generator, Optional
from src.config import PINECONE_CONFIG, TEXT_PROCESSING_CONFIG

from src.cache_manager import get_cache_key, save_to_cache, load_from_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from src.logger import setup_logger
from src.error_handler import handle_rag_error, DataSourceError
from src.types import (
    Chunk, EmbeddingList, Embedding, SearchResults,
    PineconeIndex, EmbeddingFunction, TopK
)

logger = setup_logger()


class PineconeInterface(ABC):
    """Abstract base class defining the interface for Pinecone operations."""
    
    @abstractmethod
    def list_indexes(self) -> List[str]:
        """List all available Pinecone indexes."""
        pass

    @abstractmethod
    def create_index(self, name: str, dimension: int, metric: str, spec: Any) -> None:
        """Create a new Pinecone index with specified parameters."""
        pass

    @abstractmethod
    def Index(self, name: str) -> PineconeIndex:
        """Get a Pinecone index instance by name."""
        pass

class BasePineconeManager(ABC):
    """Abstract base class for Pinecone management operations."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the Pinecone index is available."""
        pass

    @abstractmethod
    def upsert_embeddings(self, chunks: List[Chunk], embeddings: EmbeddingList) -> None:
        """Upsert embeddings and their corresponding chunks to Pinecone."""
        pass

    @abstractmethod
    def search_similar(self, query_embedding: Embedding, top_k: TopK = TEXT_PROCESSING_CONFIG['top_k_chunks']) -> SearchResults:
        """Search for similar vectors in Pinecone."""
        pass

    @abstractmethod
    def clear_index(self) -> None:
        """Clear all vectors from the Pinecone index."""
        pass

    @abstractmethod
    def get_or_create_embeddings(
        self, chunks: List[Chunk], 
        embedding_function: EmbeddingFunction
    ) -> EmbeddingList:
        """Get existing embeddings from cache or create new ones."""
        pass

    @abstractmethod
    def batch_operation(self) -> Generator[None, None, None]:
        """Context manager for batch operations."""
        pass

class PineconeManager(BasePineconeManager):
    """Implementation of Pinecone management operations with per-book indexes."""

    def __init__(
        self,
        project_id: str,
        pinecone_client: Optional[Pinecone] = None,
        max_retries: int = 3,
        min_wait: int = 1,
        max_wait: int = 10
    ):
        """
        Initialize PineconeManager for a specific book.

        Args:
            book_id (str): Unique identifier for the book
            pinecone_client (Optional[Pinecone]): Existing Pinecone client
            max_retries (int): Maximum number of retry attempts
            min_wait (int): Minimum wait time between retries
            max_wait (int): Maximum wait time between retries
        """
        self.project_id = project_id
        sanitized_id = ''.join(c.lower() for c in project_id if c.isalnum() or c == '-')
        self.index_name = f"{PINECONE_CONFIG['index_prefix']}{sanitized_id}"
        self.pc = pinecone_client or Pinecone(api_key=PINECONE_CONFIG['api_key'])
        self.index = None
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        # Только подключаемся к существующему индексу, если он есть
        self._connect_to_index()

    def _connect_to_index(self) -> None:
        """Try to connect to existing index."""
        try:
            indexes = self.pc.list_indexes()
            if self.index_name in indexes:
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Connected to existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {str(e)}")
            self.index = None

    def initialize_index(self) -> None:
        """Create new index if it doesn't exist."""
        if self.index is not None:
            logger.info(f"Index {self.index_name} already exists")
            return
        
        try:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=PINECONE_CONFIG['dimension'],
                metric=PINECONE_CONFIG['metric'],
                spec=ServerlessSpec(
                    cloud=PINECONE_CONFIG['cloud'],
                    region=PINECONE_CONFIG['region']
                )
            )
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully created Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {str(e)}")
            raise DataSourceError(f"Failed to create Pinecone index: {str(e)}")

    def _pinecone_query(self, vector: List[float], top_k: int, include_metadata: bool = True) -> Dict[str, Any]:
        try:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
        except ValueError as e:
            if "argument order" in str(e):
                # Fallback to old API if the new one fails
                return self.index.query(vector, top_k=top_k, include_metadata=include_metadata)
            raise

    def is_available(self) -> bool:
        return self.index is not None

    def upsert_embeddings(self, chunks: List[Chunk], embeddings: EmbeddingList) -> None:
        if not self.is_available():
            raise ValueError("Pinecone index is not initialized")
        
        vectors = [(str(i), embedding, {"chunk": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        self.index.upsert(vectors=vectors)

    def search_similar(self, query_embedding: Embedding, top_k: int = TEXT_PROCESSING_CONFIG['top_k_chunks']) -> SearchResults:
        if not self.is_available():
            raise ValueError("Pinecone index is not initialized")
        
        results = self._pinecone_query(query_embedding, top_k=top_k, include_metadata=True)
        return [{"chunk": match['metadata'].get('chunk', ''), "score": match['score']} for match in results['matches']]

    def clear_index(self) -> None:
        if self.is_available():
            self.index.delete(delete_all=True)
        else:
            logger.warning("Pinecone index is not available. Skipping clear operation.")

    def get_or_create_embeddings(self, chunks: List[Chunk], embedding_function: EmbeddingFunction) -> Generator[Embedding, None, None]:
        """Return generator instead of list to avoid memory accumulation"""
        for chunk in chunks:
            cache_key = get_cache_key(chunk)
            cached_embedding = load_from_cache(cache_key)
            
            if cached_embedding is not None:
                yield cached_embedding
            else:
                embedding = embedding_function([chunk])[0]
                save_to_cache(cache_key, embedding)
                yield embedding

    @contextmanager
    def batch_operation(self) -> Generator[None, None, None]:
        # This is a placeholder for potential batch operations
        # In a real implementation, you might start a transaction or prepare a batch
        yield
        # After yield, you might commit the transaction or execute the batch

    def check_pinecone_index(self):
        try:
            index_stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index stats: {index_stats}")
            return index_stats
        except Exception as e:
            logger.error(f"Error checking Pinecone index: {str(e)}", exc_info=True)
            return None

def generate_index_name(book_id: str) -> str:
    """
    Generate unique Pinecone index name for a book.
    
    Args:
        book_id (str): Unique identifier of the book (e.g., hash of content)
        
    Returns:
        str: Index name in format 'book-{book_id}'
    """
    prefix = PINECONE_CONFIG['index_prefix']
    return f"{prefix}{book_id}"
