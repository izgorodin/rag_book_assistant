from typing import List, Dict, Any, Optional, Callable, Union
import os
import numpy as np
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    BATCH_SIZE,
    OPENAI_HTTP_CONFIG
)
from src.utils.logger import setup_logger
from src.cache_manager import CacheManager
from src.utils.metrics import MetricsCollector
from src.text_processing import extract_dates, extract_named_entities, extract_key_phrases

logger = setup_logger()

class EmbeddingError(Exception):
    """Base class for embedding-related errors."""
    pass

class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding dimension is incorrect."""
    pass

class EmbeddingServiceError(EmbeddingError):
    """Raised when there's an error in the embedding service."""
    pass

class EmbeddingService:
    """Service for creating and managing embeddings with error handling and monitoring."""
    
    def __init__(
        self,
        openai_client: OpenAI,
        cache_manager: CacheManager,
        progress_callback: Optional[Callable] = None
    ):
        self.openai_client = openai_client
        self.cache_manager = cache_manager
        self.progress_callback = progress_callback
        self.batch_size = BATCH_SIZE
        logger.info("EmbeddingService initialized")

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for single text."""
        return self.create_embeddings([text])[0]

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        try:
            all_embeddings = []
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._process_batch(batch)
                all_embeddings.extend(batch_embeddings)

                if self.progress_callback:
                    current_batch = i // self.batch_size + 1
                    self.progress_callback(f"Processing embeddings batch {current_batch}/{total_batches}", 
                                        current_batch, total_batches)

            return all_embeddings
        except Exception as e:
            logger.error(f"Error in create_embeddings: {str(e)}")
            raise EmbeddingServiceError(f"Failed to create embeddings: {str(e)}")

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a batch of texts to create embeddings."""
        try:
            embeddings = []
            for text in batch:
                # Try to get from cache first
                cached_embedding = self.cache_manager.get(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue

                # If not in cache, create new embedding
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                
                embedding = response.data[0].embedding
                self.cache_manager.set(text, embedding)
                embeddings.append(embedding)

            return embeddings
        except Exception as e:
            logger.error(f"Error in _process_batch: {str(e)}")
            raise EmbeddingServiceError(f"Failed to process batch: {str(e)}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    if a.shape != b.shape:
        raise ValueError(f"Vectors have different shapes: {a.shape} and {b.shape}")
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)
