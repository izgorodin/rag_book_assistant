from typing import List, Dict, Any, Optional
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from src.config import (
    EMBEDDING_DIMENSION, 
    EMBEDDING_MODEL,
    BATCH_SIZE  # Добавим в config.py: BATCH_SIZE = 100
)
from src.logger import setup_logger
from src.text_processing import (
    extract_dates, 
    extract_named_entities, 
    extract_key_phrases
)
from src.book_data_interface import BookDataInterface
from src.pinecone_manager import PineconeManager
from src.cache_manager import CacheManager

logger = setup_logger()

class EmbeddingService:
    """Service for creating and managing embeddings."""
    
    def __init__(
        self, 
        openai_client: OpenAI,
        cache_manager: CacheManager,
        batch_size: int = BATCH_SIZE
    ):
        self.client = openai_client
        self.cache_manager = cache_manager
        self.batch_size = batch_size

    def _get_cached_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from cache if they exist."""
        cached_embeddings = []
        cache_hits = True
        
        for text in texts:
            cached = self.cache_manager.load(text)
            if cached is None:
                cache_hits = False
                break
            cached_embeddings.append(cached)
            
        return cached_embeddings if cache_hits else None

    def _cache_embeddings(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Cache embeddings for given texts."""
        for text, embedding in zip(texts, embeddings):
            self.cache_manager.save(text, embedding)

    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Create embeddings for chunks in batches."""
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + self.batch_size]
            
            # Проверяем кэш
            cached_embeddings = self._get_cached_embeddings(batch)
            if cached_embeddings:
                logger.info(f"Using cached embeddings for batch {i//self.batch_size + 1}")
                all_embeddings.extend(cached_embeddings)
                continue
            
            # Создаем новые эмбеддинги
            response = self.client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            
            # Кэшируем результаты
            self._cache_embeddings(batch, batch_embeddings)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        if not text.strip():
            return [0.0] * EMBEDDING_DIMENSION
            
        cached_embedding = self.cache_manager.load(text)
        if cached_embedding:
            return cached_embedding

        response = self.client.embeddings.create(
            input=[text], 
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        
        if len(embedding) != EMBEDDING_DIMENSION:
            logger.error(f"Incorrect embedding dimension: {len(embedding)}")
            raise ValueError(f"Incorrect embedding dimension: {len(embedding)}")
            
        self.cache_manager.save(text, embedding)
        return embedding

def create_book_data(
    preprocessed_data: Dict[str, Any],
    embedding_service: EmbeddingService
) -> BookDataInterface:
    """Create BookDataInterface instance with embeddings and processed text."""
    chunks = preprocessed_data.get('chunks', [])
    if not chunks:
        logger.error("No chunks found in preprocessed data")
        raise ValueError("No chunks found in preprocessed data")
        
    logger.info(f"Creating embeddings for {len(chunks)} chunks")
    embeddings = embedding_service.create_embeddings(chunks)
    
    processed_text = {
        'dates': preprocessed_data.get('dates', []),
        'entities': preprocessed_data.get('entities', {}),
        'key_phrases': preprocessed_data.get('key_phrases', [])
    }
    
    return BookDataInterface(chunks, embeddings, processed_text, embedding_service)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    if a.shape != b.shape:
        raise ValueError(f"Vectors have different shapes: {a.shape} and {b.shape}")
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
