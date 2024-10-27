from typing import Dict, Any
from logger import setup_logger
from book_data_interface import BookDataInterface
from embedding import EmbeddingService
from text_processing import load_and_preprocess_text
from vector_store_service import VectorStoreService

logger = setup_logger()

class BookDataFactory:
    def __init__(self, embedding_service: EmbeddingService, 
                 vector_store_service: VectorStoreService):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service

    def create_from_text(self, text: str) -> BookDataInterface:
        """Create BookDataInterface from raw text."""
        # Предобработка текста
        preprocessed_data = load_and_preprocess_text(text)
        chunks = preprocessed_data.get('chunks', [])
        
        if not chunks:
            raise ValueError("No chunks found in preprocessed data")
            
        # Создаем эмбеддинги
        embeddings = self.embedding_service.create_embeddings(chunks)
        
        # Сохраняем в vector store
        self.vector_store_service.store_vectors(chunks, embeddings)
        
        # Создаем интерфейс книги
        return BookDataInterface(
            chunks=chunks,
            embeddings=embeddings,
            processed_text=preprocessed_data
        )
