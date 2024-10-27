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
        """Create embeddings for multiple chunks of text in batches."""
        if not chunks:
            return []
        
        # Ensure chunks is a list
        if not isinstance(chunks, list):
            logger.error(f"Expected list of strings, got {type(chunks)}")
            chunks = list(chunks)
            
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        embeddings = []
        
        # Process chunks in batches with progress bar
        with tqdm(total=len(chunks), desc="Creating embeddings") as pbar:
            for i in range(0, len(chunks), self.batch_size):
                batch = list(chunks[i:i + self.batch_size])  # Ensure batch is a list
                # Skip empty chunks
                batch = [chunk for chunk in batch if chunk.strip()]
                if not batch:
                    continue
                
                # Get cached embeddings
                batch_embeddings = []
                uncached_chunks = []
                uncached_indices = []
                
                for j, chunk in enumerate(batch):
                    cached_embedding = self.cache_manager.load(chunk)
                    if cached_embedding is not None:
                        batch_embeddings.append(cached_embedding)
                    else:
                        uncached_chunks.append(chunk)
                        uncached_indices.append(j)
                
                # Create embeddings for uncached chunks
                if uncached_chunks:
                    response = self.client.embeddings.create(
                        input=uncached_chunks,
                        model=EMBEDDING_MODEL
                    )
                    
                    # Insert new embeddings at correct positions
                    for idx, emb_data in zip(uncached_indices, response.data):
                        embedding = emb_data.embedding
                        if len(embedding) != EMBEDDING_DIMENSION:
                            raise ValueError(f"Incorrect embedding dimension: {len(embedding)}")
                        
                        batch_embeddings.insert(idx, embedding)
                        # Cache the new embedding
                        self.cache_manager.save(uncached_chunks[uncached_indices.index(idx)], embedding)
                
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        # Store vectors in Pinecone
        if embeddings:
            vectors = [
                {
                    "id": str(i),
                    "values": emb,
                    "metadata": {"text": chunk}
                }
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                if chunk.strip()
            ]
            self.vector_store.upsert_vectors(vectors)
            logger.info(f"Stored {len(vectors)} vectors in Pinecone")
            
        return embeddings

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
