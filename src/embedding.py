from typing import List, Tuple, Dict, Any
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
import logging
from src.cache_manager import get_cache_key, save_to_cache, load_from_cache
from src.text_processing import extract_dates, extract_named_entities, extract_key_phrases
from src.book_data_interface import BookDataInterface

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Create embeddings for chunks in batches.
    
    Args:
        chunks (List[str]): List of text chunks to embed.
        batch_size (int): Number of chunks to process in each batch.
    
    Returns:
        List[List[float]]: List of embeddings for all chunks.
    """
    all_embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
        batch = chunks[i:i+batch_size]
        cache_keys = [get_cache_key(chunk) for chunk in batch]
        cached_embeddings = [load_from_cache(key) for key in cache_keys]
        
        # Find chunks that need embedding
        new_chunks = [chunk for chunk, emb in zip(batch, cached_embeddings) if emb is None]
        if new_chunks:
            response = client.embeddings.create(input=new_chunks, model=EMBEDDING_MODEL)
            new_embeddings = [item.embedding for item in response.data]
            
            # Save new embeddings to cache
            for chunk, emb in zip(new_chunks, new_embeddings):
                save_to_cache(get_cache_key(chunk), emb)
            
            # Merge cached and new embeddings
            all_embeddings.extend([emb if emb is not None else new_embeddings.pop(0) for emb in cached_embeddings])
        else:
            all_embeddings.extend(cached_embeddings)
    
    return all_embeddings

def save_chunks_and_embeddings(book_data: BookDataInterface, file_path: str):
    logger.info(f"Saving {len(book_data.chunks)} chunks, their embeddings, and processed text to {file_path}.")
    with open(file_path, 'wb') as f:
        pickle.dump(book_data, f)
    logger.info(f"Data saved to {file_path}.")

def load_chunks_and_embeddings(file_path: str) -> Tuple[List[str], List[List[float]], Dict[str, Any]]:
    logger.info(f"Loading chunks, embeddings, and processed text from {file_path}.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data['chunks'])} chunks, embeddings, and processed text.")
    return data['chunks'], data.get('embeddings', []), data.get('processed_text', {})

def get_or_create_chunks_and_embeddings(chunks: List[str], cache_file: str) -> BookDataInterface:
    if os.path.exists(cache_file):
        logger.info(f"Loading chunks, embeddings, and processed text from {cache_file}.")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, BookDataInterface):
            logger.info(f"Loaded {len(data.chunks)} chunks, embeddings, and processed text.")
            return data
        else:
            logger.warning("Cached data is not in the correct format. Recreating embeddings and processed text.")

    embeddings = create_embeddings(chunks)
    full_text = ' '.join(chunks)
    processed_text = {
        'dates': extract_dates(full_text),
        'entities': extract_named_entities(full_text),
        'key_phrases': extract_key_phrases(full_text)
    }
    book_data = BookDataInterface(chunks, embeddings, processed_text)
    save_chunks_and_embeddings(book_data, cache_file)
    
    return book_data

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_or_create_query_embedding(query: str, cache_file: str) -> List[float]:
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        if query in cache:
            return cache[query]
    
    embedding = create_embeddings([query])[0]
    
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    
    cache[query] = embedding
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    return embedding
