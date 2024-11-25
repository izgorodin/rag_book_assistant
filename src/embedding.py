from typing import List, Optional, Callable  # Import necessary types for type hinting
import numpy as np  # Import NumPy for numerical operations
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError  # Import OpenAI client and error classes
from src.config import (
    BATCH_SETTINGS,
    BATCH_SIZES,
    EMBEDDING_MODEL,  # Import the embedding model configuration
    PINECONE_BATCH_SIZE  # Import the batch size configuration for Pinecone
)
from src.utils.logger import get_main_logger, get_rag_logger  # Import logging utilities
from src.cache_manager import CacheManager  # Import cache manager for caching embeddings
from src.utils.metrics import MetricsCollector  # Import metrics collector for monitoring
from src.services.batch_processor import BatchProcessor  # Import batch processor for batched processing

logger = get_main_logger()  # Initialize the main logger
rag_logger = get_rag_logger()  # Initialize the RAG logger

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
        openai_client: OpenAI,  # OpenAI client for generating embeddings
        cache_manager: CacheManager,  # Cache manager for storing embeddings
        metrics_collector: Optional[MetricsCollector] = None,  # Optional metrics collector for monitoring
        progress_callback: Optional[Callable] = None,  # Optional callback for progress updates
        batch_size: int = PINECONE_BATCH_SIZE  # Batch size for processing embeddings
    ):
        self.client = openai_client  # Assign OpenAI client to instance variable
        self.cache_manager = cache_manager  # Assign cache manager to instance variable
        self.metrics = metrics_collector  # Assign metrics collector to instance variable
        self.progress_callback = progress_callback  # Assign progress callback to instance variab
        self.batch_size = batch_size  # Assign batch size to instance variable
        self.logger = get_main_logger()  # Добавляем инициализацию логгера
        self.rag_logger = get_rag_logger()  # И RAG логгера если нужен
        self.batch_processor = BatchProcessor(
            batch_size=BATCH_SIZES['embeddings'],
            max_workers=BATCH_SETTINGS['max_workers']
        )
        logger.info("EmbeddingService initialized")  # Log initialization of the service
        rag_logger.info("\nEmbedding Service:\nStatus: Initialized\n" + "-"*50)  # Log status in RAG logger

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using batched processing."""
        self.logger.info(f"Creating embeddings for {len(texts)} texts")
        
        async def process_batch(batch: List[str]) -> List[List[float]]:
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                return [emb_data.embedding for emb_data in response.data]
            except Exception as e:
                self.logger.error(f"Error creating embeddings: {str(e)}")
                raise

        return await self.batch_processor.process_async(
            items=texts,
            processor=process_batch,
            description="Creating embeddings"
        )

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a batch of texts to create embeddings with caching."""
        embeddings = []  # List to store embeddings for the current batch
        uncached_texts = []  # List to store texts that are not cached
        uncached_indices = []  # List to store indices of uncached texts

        # Check the cache for each text in the batch
        for i, text in enumerate(batch):
            cached = self.cache_manager.get(text)  # Attempt to retrieve cached embedding
            if cached is not None:  # If a cached embedding is found
                embeddings.append(cached)  # Add cached embedding to the list
            else:  # If no cached embedding is found
                uncached_texts.append(text)  # Add text to uncached list
                uncached_indices.append(i)  # Store index of uncached text

        # Create embeddings for uncached texts
        if uncached_texts:
            response = self.client.embeddings.create(
                input=uncached_texts,  # Input the uncached texts
                model=EMBEDDING_MODEL  # Specify the embedding model
            )
            
            for i, emb_data in enumerate(response.data):  # Iterate over the response data
                embedding = emb_data.embedding  # Extract the embedding
                
                # Cache the new embedding
                self.cache_manager.set(uncached_texts[i], embedding)  # Store the embedding in cache
                embeddings.insert(uncached_indices[i], embedding)  # Insert the embedding at the correct index

        return embeddings  # Return the list of embeddings for the batch

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64)  # Convert vector a to a NumPy array
    b = np.array(b, dtype=np.float64)  # Convert vector b to a NumPy array
    
    if a.shape != b.shape:  # Check if the shapes of the vectors are different
        raise ValueError(f"Vectors have different shapes: {a.shape} and {b.shape}")  # Raise an error if shapes differ
    
    norm_a = np.linalg.norm(a)  # Calculate the norm of vector a
    norm_b = np.linalg.norm(b)  # Calculate the norm of vector b
    
    if norm_a == 0 or norm_b == 0:  # Check for zero vectors
        return 0.0  # Return 0.0 for cosine similarity if either vector is zero
        
    return np.dot(a, b) / (norm_a * norm_b)  # Calculate and return cosine similarity
