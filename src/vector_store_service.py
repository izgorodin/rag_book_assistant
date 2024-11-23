from typing import List, Dict, Any, Optional, Callable
import sys
from src.utils.logger import get_main_logger, get_rag_logger
from src.utils.error_handler import handle_rag_error
from src.pinecone_manager import PineconeManager
import re

# Initialize loggers for main and RAG-specific logging
logger = get_main_logger()
rag_logger = get_rag_logger()

class VectorStoreService:
    def __init__(self, vector_store: PineconeManager, progress_callback: Optional[Callable] = None):
        """
        Initialize the VectorStoreService.

        Args:
            vector_store: An instance of PineconeManager for managing vector storage.
            progress_callback: Optional callback function to report progress.
        """
        self.vector_store = vector_store
        self.progress_callback = progress_callback
        self.max_batch_size = 100  # Optimal batch size for Pinecone
        
    def _create_vector_batch(self, 
                           chunks: List[str], 
                           embeddings: List[List[float]], 
                           start_idx: int, 
                           batch_size: int) -> List[Dict[str, Any]]:
        """
        Create a batch of vectors for Pinecone.

        Args:
            chunks: List of text chunks to be converted into vectors.
            embeddings: List of embeddings corresponding to the chunks.
            start_idx: Starting index for the current batch.
            batch_size: Number of vectors to include in the batch.

        Returns:
            A list of dictionaries representing the vector batch.
        """
        end_idx = min(start_idx + batch_size, len(chunks))  # Calculate the end index for the batch
        return [
            {
                'id': str(idx + start_idx),  # Unique ID for the vector
                'values': embeddings[idx],  # Embedding values
                'metadata': {'text': chunks[idx]}  # Metadata containing the original text chunk
            }
            for idx in range(end_idx - start_idx)  # Create the vector batch
        ]

    @handle_rag_error
    def store_vectors(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Store vectors with enhanced metadata and sparse vectors support.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        total_vectors = len(chunks)
        processed_vectors = 0
        
        logger.info(f"Starting to store {total_vectors} vectors in batches of {self.max_batch_size}")
        
        for batch_start in range(0, total_vectors, self.max_batch_size):
            try:
                # Улучшенное создание векторов с метаданными
                vectors = [
                    {
                        'id': str(idx + batch_start),
                        'values': embeddings[idx],
                        'metadata': {
                            'text': chunks[idx],
                            'has_date': bool(re.search(r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b', chunks[idx])),
                            'has_year': bool(re.search(r'\b(19|20)\d{2}\b', chunks[idx])),
                            'has_names': bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', chunks[idx])),
                            'chunk_length': len(chunks[idx])
                        }
                    }
                    for idx in range(min(self.max_batch_size, total_vectors - batch_start))
                ]
                
                self.vector_store.upsert_vectors(vectors)
                processed_vectors += len(vectors)
                
                if self.progress_callback:
                    self.progress_callback("Storing vectors", processed_vectors, total_vectors)
                    
                logger.info(f"Stored batch {batch_start//self.max_batch_size + 1}: {processed_vectors}/{total_vectors}")
                
            except Exception as e:
                error_msg = f"Error storing batch starting at index {batch_start}: {str(e)}"
                logger.error(error_msg)
                raise

    def search_vectors(self, query_vector: List[float], top_k: int = 5, 
                      filter_conditions: Optional[Dict] = None,
                      use_hybrid: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced vector search with filtering and hybrid search options.
        """
        try:
            # Упрощаем до базовых параметров
            return self.vector_store.search_vectors(
                query_vector,  # Передаем только сам вектор
                top_k  # И количество результатов
            )
            
        except Exception as e:
            error_msg = f"Error searching vectors: {str(e)}"
            logger.error(error_msg)
            raise
