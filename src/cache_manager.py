from abc import ABC, abstractmethod
import os
import shutil
import hashlib
import pickle
from typing import Any, Dict, Optional
from src.config import CACHE_DIR
from src.logger import setup_logger

logger = setup_logger()

class CacheInterface(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """Save data to cache."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

class FileSystemCache(CacheInterface):
    """File system based cache implementation."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_file_path(self, key: str) -> str:
        """Generate file path for cache key."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")
    
    def save(self, key: str, data: Any) -> None:
        """Save data to cache file."""
        try:
            file_path = self._get_file_path(key)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved data to cache: {key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            raise
    
    def load(self, key: str) -> Optional[Any]:
        """Load data from cache file."""
        try:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded data from cache: {key}")
                return data
            return None
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return None
    
    def clear(self) -> None:
        """Clear all cache files."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return os.path.exists(self._get_file_path(key))

class CacheManager:
    """Manages caching operations with file change detection."""
    
    WATCHED_FILES = [
        'src/cli.py',
        'src/rag.py',
        'src/embedding.py'
    ]
    
    def __init__(
        self,
        cache_implementation: CacheInterface,
        hash_file: str = 'file_hashes.txt'
    ):
        self.cache = cache_implementation
        self.hash_file = hash_file
        self._check_for_changes()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _save_file_hashes(self) -> None:
        """Save current file hashes."""
        hashes = {}
        for file in self.WATCHED_FILES:
            if os.path.exists(file):
                hashes[file] = self._get_file_hash(file)
        
        with open(self.hash_file, 'w') as f:
            for file, hash_value in hashes.items():
                f.write(f"{file}:{hash_value}\n")
    
    def _check_for_changes(self) -> None:
        """Check for changes in watched files and clear cache if needed."""
        if not os.path.exists(self.hash_file):
            self.cache.clear()
            self._save_file_hashes()
            return
        
        with open(self.hash_file, 'r') as f:
            stored_hashes = dict(line.strip().split(':') for line in f)
        
        for file in self.WATCHED_FILES:
            if os.path.exists(file):
                current_hash = self._get_file_hash(file)
                if file not in stored_hashes or stored_hashes[file] != current_hash:
                    logger.info(f"Detected changes in {file}")
                    self.cache.clear()
                    self._save_file_hashes()
                    return
    
    def save(self, key: str, data: Any) -> None:
        """Save data to cache."""
        self.cache.save(key, data)
    
    def load(self, key: str) -> Optional[Any]:
        """Load data from cache."""
        return self.cache.load(key)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.cache.exists(key)

# Create default cache manager instance
default_cache = CacheManager(FileSystemCache(CACHE_DIR))
