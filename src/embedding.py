from typing import List, Optional, Callable
import numpy as np
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from src.config import (
    EMBEDDING_MODEL,
    PINECONE_BATCH_SIZE
)
from src.utils.logger import setup_logger
from src.cache_manager import CacheManager
from src.utils.metrics import MetricsCollector

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
        metrics_collector: Optional[MetricsCollector] = None,
        progress_callback: Optional[Callable] = None,
        batch_size: int = PINECONE_BATCH_SIZE
    ):
        self.client = openai_client
        self.cache_manager = cache_manager
        self.metrics = metrics_collector
        self.progress_callback = progress_callback
        self.batch_size = batch_size
        logger.info("EmbeddingService initialized")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts with batching and caching."""
        try:
            all_embeddings = []
            total_texts = len(texts)
            processed_texts = 0

            # Создаем батчи
            for i in range(0, total_texts, self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._process_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Обновляем прогресс
                processed_texts += len(batch)
                if self.progress_callback:
                    self.progress_callback(
                        "Creating embeddings",
                        processed_texts,
                        total_texts
                    )

            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in create_embeddings: {str(e)}")
            raise

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a batch of texts to create embeddings with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Проверяем кэш
        for i, text in enumerate(batch):
            cached = self.cache_manager.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Создаем эмбеддинги для некэшированных текстов
        if uncached_texts:
            response = self.client.embeddings.create(
                input=uncached_texts,
                model=EMBEDDING_MODEL
            )
            
            for i, emb_data in enumerate(response.data):
                embedding = emb_data.embedding
                
                # Кэшируем новый эмбеддинг
                self.cache_manager.set(uncached_texts[i], embedding)
                embeddings.insert(uncached_indices[i], embedding)

        return embeddings

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
