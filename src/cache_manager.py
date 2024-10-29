from abc import ABC, abstractmethod  # Importing abstract base class and abstract method for interface definition
import os  # Importing os module for file and directory operations
import pickle  # Importing pickle for object serialization
import hashlib  # Importing hashlib for generating hash values
from typing import Any, Dict, Optional  # Importing typing for type hinting
from src.config import CACHE_DIR  # Importing cache directory configuration
from src.utils.logger import get_main_logger, get_rag_logger  # Importing logging utilities

logger = get_main_logger()  # Initializing the main logger
rag_logger = get_rag_logger()  # Initializing the RAG logger

class CacheInterface(ABC):
    """Interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item in cache with optional TTL (Time To Live)."""
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
        """Get cache statistics such as hits and misses."""
        pass

class CacheManager(CacheInterface):
    """File system based cache implementation with statistics."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls, cache_dir: str):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: str):
        """Initialize CacheManager with a specified cache directory."""
        if not self._initialized:
            self.cache_dir = cache_dir
            self.hits = 0
            self.misses = 0
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"CacheManager initialized at {cache_dir}")
            self._initialized = True
            self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization details."""
        rag_logger.info(
            f"\nCache Initialization:\n"
            f"Directory: {self.cache_dir}\n"
            f"Status: Ready\n"
            f"{'-'*50}"
        )
    
    def _get_path(self, key: str) -> str:
        """Generate file path for cache key based on its hash."""
        hashed = hashlib.sha256(key.encode()).hexdigest()  # Create a hash of the key
        return os.path.join(self.cache_dir, f"{hashed}.cache")  # Return the full path for the cache file

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache using the specified key."""
        try:
            path = self._get_path(key)  # Get the cache file path
            if os.path.exists(path):  # Check if the cache file exists
                with open(path, 'rb') as f:  # Open the cache file for reading
                    value = pickle.load(f)  # Load the cached value
                    self.hits += 1  # Increment hits counter
                    logger.debug(f"Cache hit for key: {key}")  # Log cache hit
                    return value  # Return the cached value
            self.misses += 1  # Increment misses counter
            logger.debug(f"Cache miss for key: {key}")  # Log cache miss
        except Exception as e:
            error_msg = f"Error reading from cache: {str(e)}"  # Prepare error message
            logger.error(error_msg)  # Log the error
            rag_logger.error(f"\nCache Error:\n{error_msg}\n{'-'*50}")  # Log error in RAG logger
            self.misses += 1  # Increment misses counter
        return None  # Return None if not found

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store an item in the cache with an optional TTL."""
        try:
            path = self._get_path(key)  # Get the cache file path
            with open(path, 'wb') as f:  # Open the cache file for writing
                pickle.dump(value, f)  # Serialize and store the value
            logger.debug(f"Cache set for key: {key}")  # Log successful cache set
        except Exception as e:
            error_msg = f"Error writing to cache: {str(e)}"  # Prepare error message
            logger.error(error_msg)  # Log the error
            rag_logger.error(f"\nCache Error:\n{error_msg}\n{'-'*50}")  # Log error in RAG logger

    def delete(self, key: str) -> bool:
        """Remove an item from the cache using the specified key."""
        try:
            path = self._get_path(key)  # Get the cache file path
            if os.path.exists(path):  # Check if the cache file exists
                os.remove(path)  # Remove the cache file
                return True  # Return True if deletion was successful
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")  # Log deletion error
        return False  # Return False if deletion failed

    def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            for file in os.listdir(self.cache_dir):  # Iterate through files in cache directory
                if file.endswith('.cache'):  # Check for cache files
                    os.remove(os.path.join(self.cache_dir, file))  # Remove the cache file
            self.hits = 0  # Reset hits counter
            self.misses = 0  # Reset misses counter
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")  # Log clearing error

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including hits, misses, and hit ratio."""
        total_ops = self.hits + self.misses  # Calculate total operations
        return {
            "hits": self.hits,  # Return hits count
            "misses": self.misses,  # Return misses count
            "hit_ratio": self.hits / total_ops if total_ops > 0 else 0  # Calculate and return hit ratio
        }

# Create default cache instance using singleton pattern
default_cache = CacheManager(CACHE_DIR)
