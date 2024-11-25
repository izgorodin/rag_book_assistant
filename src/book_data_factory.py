from typing import Dict, Any, Optional, Callable, List, Union
from src.utils.logger import get_main_logger, get_rag_logger
from src.book_data_interface import BookDataInterface
from src.embedding import EmbeddingService
from src.services.text_processor import TextProcessor
import time
import asyncio
import uuid
from src.vector_store_service import VectorStoreService

logger = get_main_logger()
rag_logger = get_rag_logger()

class BookDataFactory:
    def __init__(self, 
                 vector_store_service: VectorStoreService,
                 embedding_service: EmbeddingService):
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self.logger = get_main_logger()

    async def create_from_text(self, text: str) -> BookDataInterface:
        """Create BookData from text content."""
        try:
            # Создаем уникальный идентификатор документа как namespace
            namespace = str(uuid.uuid4())
            
            # Инициализируем TextProcessor с неймспейсом
            text_processor = TextProcessor(namespace=namespace)
            
            # Получаем обработанные данные через TextProcessor
            processed_data = await text_processor.process_text(text)
            
            # Создаем эмбеддинги для чанков
            embeddings = await self.embedding_service.create_embeddings(
                processed_data['chunks']
            )
            
            # Сохраняем в векторное хранилище
            await self._store_vectors(
                chunks=processed_data['chunks'],
                embeddings=embeddings,
                namespace=namespace,
                metadata=processed_data['metadata']
            )

            return BookDataInterface(
                namespace=namespace,
                chunks=processed_data['chunks'],
                embeddings=embeddings,
                processed_text=processed_data,
                embedding_service=self.embedding_service,
                vector_store_service=self.vector_store_service,
                metadata=processed_data['metadata']
            )
            
        except Exception as e:
            self.logger.error(f"Error creating book data: {str(e)}")
            raise

    async def _store_vectors(self, 
                           chunks: List[str], 
                           embeddings: List[List[float]], 
                           namespace: str, 
                           metadata: Dict[str, Any]) -> None:
        """Store vectors in the vector store"""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vector_metadata = {
                'text': chunk,
                'namespace': namespace
            }
            # Добавляем остальные метаданные как плоские значения
            vector_metadata.update({
                k: str(v) if isinstance(v, (list, dict)) else v
                for k, v in metadata.items()
            })
            
            vectors.append({
                'values': emb,
                'metadata': vector_metadata
            })
            
        await self.vector_store_service.store_vectors(vectors)

