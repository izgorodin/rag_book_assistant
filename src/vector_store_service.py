from typing import List, Dict, Any, Optional, Callable
from src.config import TOP_K_CHUNKS
from src.utils.logger import get_main_logger, get_rag_logger
from src.utils.error_handler import handle_rag_error
import re
from typing import List, Dict, Any, Optional
from src.interfaces.vector_store import VectorStore
from src.embedding import EmbeddingService

# Initialize loggers for main and RAG-specific logging
logger = get_main_logger()
rag_logger = get_rag_logger()

class VectorStoreService:
    """Service for managing vector operations."""
    
    def __init__(self, vector_store_service: VectorStore, embedding_service: Optional[EmbeddingService] = None):
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self._initialized = False
        self.logger = get_main_logger()

    async def initialize(self) -> None:
        """Initialize the vector store service."""
        if not self._initialized:
            if not await self.vector_store_service.is_available():
                await self.vector_store_service.initialize()
                if not await self.vector_store_service.is_available():
                    raise ValueError("Vector store initialization failed")
            self._initialized = True
            self.logger.info("VectorStoreService initialized")

    async def store_vectors(self, vectors: List[Dict[str, Any]], namespace: str = None) -> None:
        """Store vectors in the vector store with namespace support"""
        try:
            # Добавляем namespace в метаданные каждого вектора
            for vector in vectors:
                if 'metadata' not in vector:
                    vector['metadata'] = {}
                vector['metadata']['namespace'] = namespace
            
            # Вызываем метод store_vectors у PineconeManager только с vectors
            await self.vector_store_service.store_vectors(vectors)
        except Exception as e:
            self.logger.error(f"Error storing vectors: {str(e)}")
            raise

    @staticmethod
    def prepare_vectors(chunks: List[str], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Prepare vectors for storage from chunks and embeddings.
        """
        return [
            {
                'values': embedding,
                'metadata': {'text': chunk}
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

    async def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for query text."""
        if not self.embedding_service:
            raise ValueError("Embedding service not initialized")
        
        self.logger.info(f"Creating embedding for query: {query}")
        embeddings = await self.embedding_service.create_embeddings([query])
        self.logger.info("Query embedding created successfully")
        return embeddings[0]

    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = TOP_K_CHUNKS,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        await self.initialize()
        return await self.vector_store_service.search_vectors(
            query_vector, 
            top_k=top_k,
            filter_conditions=filter_conditions
        )

    def _create_metadata(self, text: str) -> Dict[str, Any]:
        """Create enhanced metadata for text chunk."""
        return {
            'text': text,
            'has_date': bool(re.search(r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b', text)),
            'has_year': bool(re.search(r'\b(19|20)\d{2}\b', text)),
            'has_names': bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)),
            'chunk_length': len(text)
        }