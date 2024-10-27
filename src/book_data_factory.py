from typing import Dict, Any, Optional, Callable
from logger import setup_logger
from book_data_interface import BookDataInterface
from embedding import EmbeddingService
from text_processing import load_and_preprocess_text
from vector_store_service import VectorStoreService
from tqdm import tqdm

logger = setup_logger()

class BookDataFactory:
    def __init__(self, 
                 embedding_service: EmbeddingService, 
                 vector_store_service: VectorStoreService,
                 progress_callback: Optional[Callable[[str, int, int], None]] = None):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.progress_callback = progress_callback

    def create_from_text(self, text: str) -> BookDataInterface:
        """Create BookDataInterface from raw text."""
        if self.progress_callback:
            self.progress_callback("Starting text processing", 0, 3)
            
        # Предобработка текста
        preprocessed_data = load_and_preprocess_text(text)
        chunks = preprocessed_data.get('chunks', [])
        
        if not chunks:
            raise ValueError("No chunks found in preprocessed data")
        
        if self.progress_callback:
            self.progress_callback("Creating embeddings", 1, 3)
            
        # Создаем эмбеддинги
        embeddings = self.embedding_service.create_embeddings(chunks)
        
        if self.progress_callback:
            self.progress_callback("Storing vectors", 2, 3)
            
        # Сохраняем в vector store
        self.vector_store_service.store_vectors(chunks, embeddings)
        
        if self.progress_callback:
            self.progress_callback("Completed", 3, 3)
            
        return BookDataInterface(
            chunks=chunks,
            embeddings=embeddings,
            processed_text=preprocessed_data
        )
