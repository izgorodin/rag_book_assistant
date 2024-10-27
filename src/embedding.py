from typing import List, Dict, Any
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
        vector_store: PineconeManager,
        cache_manager: CacheManager,
        batch_size: int = BATCH_SIZE
    ):
        self.client = openai_client
        self.vector_store = vector_store
        self.cache_manager = cache_manager
        self.batch_size = batch_size

    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Create embeddings for chunks in batches."""
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        all_embeddings = []
        batch_size = 100  # Можно настроить в зависимости от размера чанков
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = []
            
            # Создаем эмбеддинги для батча
            response = self.client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            
            # Сохраняем в векторное хранилище
            vectors = [
                {
                    'id': str(i + j),
                    'values': emb,
                    'metadata': {'text': chunk}
                }
                for j, (chunk, emb) in enumerate(zip(batch, batch_embeddings))
            ]
            
            try:
                self.vector_store.upsert_vectors(vectors)
                logger.info(f"Successfully upserted batch of {len(vectors)} vectors")
            except Exception as e:
                logger.error(f"Error upserting vectors batch: {str(e)}")
                raise
                
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
