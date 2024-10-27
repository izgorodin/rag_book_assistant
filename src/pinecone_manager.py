from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from src.config import (
    PINECONE_API_KEY, PINECONE_CLOUD, EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME, PINECONE_METRIC, PINECONE_REGION
)
from src.error_handler import RAGError
from src.logger import setup_logger

logger = setup_logger()

class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the vector store is available."""
        pass

    @abstractmethod
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Store vectors with their metadata."""
        pass

    @abstractmethod
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored vectors."""
        pass

class PineconeManager(VectorStore):
    """Manages interactions with Pinecone vector database."""
    
    def __init__(self, lazy_init=True):
        """Initialize Pinecone client and index."""
        self.initialized = False
        if not lazy_init:
            self._init()

    def _init(self):
        if self.initialized:
            return
        
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self._initialize_index()
            self.initialized = True
            logger.info("Pinecone manager initialized successfully")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
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

    def is_available(self) -> bool:
        """Check if Pinecone index is available."""
        try:
            return hasattr(self, 'index') and self.index is not None
        except Exception:
            return False

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Store vectors in Pinecone.
        
        Args:
            vectors: List of dictionaries containing vector data
        """
        if not self.is_available():
            raise ValueError("Pinecone index not initialized")
            
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            
        Returns:
            List of similar vectors with their metadata
        """
        if not self.is_available():
            raise ValueError("Pinecone index not initialized")
        
        try:
            results = self.index.query(
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
