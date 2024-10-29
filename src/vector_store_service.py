from typing import List, Dict, Any, Optional, Callable
import sys
from src.utils.logger import setup_logger
from src.utils.error_handler import handle_rag_error
from src.pinecone_manager import PineconeManager

logger = setup_logger()

class VectorStoreService:
    def __init__(self, vector_store: PineconeManager, progress_callback: Optional[Callable] = None):
        self.vector_store = vector_store
        self.progress_callback = progress_callback
        self.max_batch_size = 100  # Оптимальный размер для Pinecone
        
    def _create_vector_batch(self, 
                           chunks: List[str], 
                           embeddings: List[List[float]], 
                           start_idx: int, 
                           batch_size: int) -> List[Dict[str, Any]]:
        """Create a batch of vectors for Pinecone."""
        end_idx = min(start_idx + batch_size, len(chunks))
        return [
            {
                'id': str(idx + start_idx),
                'values': embeddings[idx],
                'metadata': {'text': chunks[idx]}
            }
            for idx in range(end_idx - start_idx)
        ]

    @handle_rag_error
    def store_vectors(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Store vectors in batches with optimal size.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embeddings corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        total_vectors = len(chunks)
        processed_vectors = 0
        
        logger.info(f"Starting to store {total_vectors} vectors in batches of {self.max_batch_size}")
        
        for batch_start in range(0, total_vectors, self.max_batch_size):
            try:
                # Создаем батч векторов
                vectors = self._create_vector_batch(
                    chunks, 
                    embeddings, 
                    batch_start, 
                    self.max_batch_size
                )
                
                # Сохраняем батч
                self.vector_store.upsert_vectors(vectors)
                
                # Обновляем счетчик и прогресс
                processed_vectors += len(vectors)
                if self.progress_callback:
                    self.progress_callback(
                        "Storing vectors",
                        processed_vectors,
                        total_vectors
                    )
                
                logger.info(f"Stored batch {batch_start//self.max_batch_size + 1}: "
                          f"{processed_vectors}/{total_vectors} vectors")
                
            except Exception as e:
                logger.error(f"Error storing batch starting at index {batch_start}: {str(e)}")
                raise

        logger.info(f"Successfully stored all {total_vectors} vectors")
