from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from src.config import (
    PINECONE_API_KEY, PINECONE_CLOUD, EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME, PINECONE_METRIC, PINECONE_REGION, TOP_K_CHUNKS
)
from src.utils.error_handler import RAGError
from src.utils.logger import get_main_logger, get_rag_logger
from src.interfaces.vector_store import VectorStore
import asyncio
import pinecone
import uuid

logger = get_main_logger()
rag_logger = get_rag_logger()

class PineconeManager(VectorStore):
    """Manages interactions with Pinecone vector database."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.logger = get_main_logger()
        self.rag_logger = get_rag_logger()
        self._index = None
        self._initialized = False

    async def is_available(self) -> bool:
        """Асинхронная проверка доступности индекса."""
        try:
            if not self._initialized:
                return False
            # Асинхронная проверка подключения
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._index.describe_index_stats()
            )
            return True
        except Exception as e:
            logger.error(f"Error checking Pinecone availability: {e}")
            return False

    async def initialize(self) -> None:
        """Асинхронная инициализация Pinecone."""
        if not self._initialized:
            try:
                # Используем новый API Pinecone
                pc = Pinecone(api_key=self.api_key)
                
                # Проверяем существование индекса
                if self.index_name not in pc.list_indexes().names():
                    logger.info(f"Creating new index: {self.index_name}")
                    pc.create_index(
                        name=self.index_name,
                        dimension=EMBEDDING_DIMENSION,
                        metric=PINECONE_METRIC,
                        spec=ServerlessSpec(
                            cloud=PINECONE_CLOUD,
                            region=PINECONE_REGION
                        )
                    )
                
                self._index = pc.Index(self.index_name)
                self._initialized = True
                logger.info("Pinecone initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                raise

    def _init(self):
        if self.initialized:
            return
        
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self._initialize_index()
            self.initialized = True
            logger.info("Pinecone manager initialized successfully")
            rag_logger.info("\nPinecone Initialization:\nStatus: Success\n" + "-"*50)
        except Exception as e:
            error_msg = f"Pinecone initialization error: {str(e)}"
            logger.error(error_msg)
            rag_logger.error(f"\nPinecone Error:\n{error_msg}\n{'-'*50}")
            raise

    def _initialize_index(self):
        """Initialize Pinecone index."""
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                self._create_index()
            else:
                logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
            
            # В любом случае подключаемся к индексу
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            logger.info("Successfully connected to Pinecone index")
            
        except Exception as e:
            if not isinstance(e, RAGError) or "ALREADY_EXISTS" not in str(e):
                logger.error(f"Error initializing Pinecone index: {str(e)}")
                raise

    def _create_index(self):
        """Create new Pinecone index."""
        try:
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            # Проверяем на ALREADY_EXISTS до логирования ошибки
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
                return
            logger.error(f"Error creating Pinecone index: {str(e)}")
            raise

    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Store vectors in Pinecone with batching"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Разбиваем векторы на батчи по 100 (чтобы уложиться в лимит 4MB)
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Подготавливаем векторы для батча
                pinecone_vectors = []
                for vector in batch:
                    vector_metadata = {
                        'text': vector['metadata']['text'][:1000],  # Ограничиваем размер текста
                        'namespace': vector['metadata'].get('namespace', '') or ''
                    }
                    # Добавляем остальные метаданные как плоские значения
                    for k, v in vector['metadata'].items():
                        if k not in ['text', 'namespace']:
                            vector_metadata[k] = str(v)[:1000] if isinstance(v, str) else str(v)
                    
                    pinecone_vectors.append({
                        'id': str(uuid.uuid4()),
                        'values': vector['values'],
                        'metadata': vector_metadata
                    })
                
                # Сохраняем батч
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._index.upsert(
                        vectors=pinecone_vectors,
                        namespace=vector_metadata['namespace']
                    )
                )
                
                self.logger.info(f"Stored batch {i//batch_size + 1}/{total_batches} ({len(pinecone_vectors)} vectors)")
                
                # Добавляем небольшую задержку между батчами
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error storing vectors in Pinecone: {str(e)}")
            raise

    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = TOP_K_CHUNKS,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        if not await self.is_available():
            raise ValueError("Pinecone index not initialized")
        
        try:
            results = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                for match in results["matches"]
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all vectors from the index."""
        if self.is_available():
            try:
                self.index.delete(delete_all=True)
                logger.info("Successfully cleared all vectors from index")
            except Exception as e:
                logger.error(f"Error clearing vectors: {str(e)}")
                raise
