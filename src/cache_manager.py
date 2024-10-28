from abc import ABC, abstractmethod
import os
import pickle
import hashlib
from typing import Any, Dict, Optional
from src.config import CACHE_DIR
from src.utils.logger import setup_logger

logger = setup_logger()

class CacheInterface(ABC):
    """Interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove item from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

class CacheManager(CacheInterface):
    """File system based cache implementation with statistics."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"CacheManager initialized at {cache_dir}")

    def _get_path(self, key: str) -> str:
        """Generate file path for cache key."""
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.cache")

    def get(self, key: str) -> Optional[Any]:
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    value = pickle.load(f)
                    self.hits += 1
                    logger.debug(f"Cache hit for key: {key}")
                    return value
            self.misses += 1
            logger.debug(f"Cache miss for key: {key}")
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")
            self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            path = self._get_path(key)
            with open(path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cache set for key: {key}")
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")

    def delete(self, key: str) -> bool:
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
                return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
        return False

    def clear(self) -> None:
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, file))
            self.hits = 0
            self.misses = 0
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        total_ops = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / total_ops if total_ops > 0 else 0
        }

# Create default cache instance
default_cache = CacheManager(CACHE_DIR)
