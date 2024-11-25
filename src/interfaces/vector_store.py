from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.config import TOP_K_CHUNKS

class VectorStore(ABC):
    """Abstract interface for vector storage implementations."""
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the store is available."""
        pass

    @abstractmethod
    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Store vectors with metadata."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = TOP_K_CHUNKS,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors."""
        pass