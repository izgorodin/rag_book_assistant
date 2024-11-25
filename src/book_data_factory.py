from typing import Dict, Any, Optional, Callable, List, Union
from src.utils.logger import get_main_logger, get_rag_logger
from src.book_data_interface import BookDataInterface
from src.embedding import EmbeddingService
from src.services.text_processor import TextProcessor
import time
import asyncio
from src.vector_store_service import VectorStoreService

logger = get_main_logger()
rag_logger = get_rag_logger()

class BookDataFactory:
    def __init__(self, 
                 vector_store_service: VectorStoreService,
                 embedding_service: EmbeddingService,
                 text_processor: Optional[TextProcessor] = None):
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self.text_processor = text_processor or TextProcessor()
        self.logger = get_main_logger()

    async def create_from_text(self, text: str) -> BookDataInterface:
        """Create BookData from text content."""
        try:
            # Разбиваем текст на чанки
            chunks = await self.text_processor.split_into_chunks(text)
            self.logger.info(f"Created {len(chunks)} chunks")

            # Создаем эмбеддинги для чанков
            embeddings = await self.embedding_service.create_embeddings(chunks)
            self.logger.info("Embeddings created successfully")

            # Получаем метаданные через TextProcessor
            processed_data = await self.text_processor.process_text(text)

            # Создаем и возвращаем BookDataInterface
            return BookDataInterface(
                chunks=chunks,
                embeddings=embeddings,
                processed_text=processed_data,
                embedding_service=self.embedding_service,
                vector_store_service=self.vector_store_service,
                dates=processed_data['metadata']['dates'],
                entities=processed_data['metadata']['entities'],
                key_phrases=processed_data['metadata']['key_phrases']
            )
        except Exception as e:
            self.logger.error(f"Error creating book data: {str(e)}")
            raise

    async def _store_vectors(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Store vectors in the vector store"""
        vectors = [
            {'values': emb, 'metadata': {'text': chunk}} 
            for emb, chunk in zip(embeddings, chunks)
        ]
        await self.vector_store_service.store_vectors(vectors)

