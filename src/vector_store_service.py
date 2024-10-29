from typing import List, Dict, Any, Optional, Callable
import sys
from src.utils.logger import get_main_logger, get_rag_logger
from src.utils.error_handler import handle_rag_error
from src.pinecone_manager import PineconeManager

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
        Store vectors in batches with optimal size.

        Args:
            chunks: List of text chunks to be stored.
            embeddings: List of embeddings corresponding to the chunks.

        Raises:
            ValueError: If the lengths of chunks and embeddings do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")  # Ensure matching lengths

        total_vectors = len(chunks)  # Total number of vectors to store
        processed_vectors = 0  # Counter for processed vectors
        
        logger.info(f"Starting to store {total_vectors} vectors in batches of {self.max_batch_size}")
        rag_logger.info(f"\nVector Storage:\nTotal vectors: {total_vectors}\nBatch size: {self.max_batch_size}\n{'-'*50}")
        
        # Process vectors in batches
        for batch_start in range(0, total_vectors, self.max_batch_size):
            try:
                vectors = self._create_vector_batch(
                    chunks, 
                    embeddings, 
                    batch_start, 
                    self.max_batch_size
                )  # Create a batch of vectors
                
                self.vector_store.upsert_vectors(vectors)  # Store the vectors in Pinecone
                
                processed_vectors += len(vectors)  # Update the count of processed vectors
                if self.progress_callback:
                    self.progress_callback(
                        "Storing vectors",  # Progress message
                        processed_vectors,
                        total_vectors
                    )
                
                logger.info(f"Stored batch {batch_start//self.max_batch_size + 1}: {processed_vectors}/{total_vectors}")
                
            except Exception as e:
                error_msg = f"Error storing batch starting at index {batch_start}: {str(e)}"
                logger.error(error_msg)  # Log the error
                rag_logger.error(f"\nVector Storage Error:\n{error_msg}\n{'-'*50}")
                raise  # Raise the exception for further handling

        success_msg = f"Successfully stored all {total_vectors} vectors"
        logger.info(success_msg)  # Log success message
        rag_logger.info(f"\n{success_msg}\n{'-'*50}")
