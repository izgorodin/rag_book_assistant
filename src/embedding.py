from typing import List, Tuple, Dict, Any, Union
import os
import pickle
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, USE_CACHING, UPDATE_EMBEDDINGS
import logging
import numpy as np
from src.text_processing import split_into_chunks
from src.cache import get_cache_key, save_to_cache, load_from_cache

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str], batch_size: int = 100, use_cache: bool = USE_CACHING) -> List[List[float]]:
    all_embeddings = []
    uncached_chunks = []
    uncached_indices = []

    if use_cache:
        # Проверяем кэш для каждого чанка
        for i, chunk in enumerate(chunks):
            cache_key = get_cache_key(chunk)
            cached_embedding = load_from_cache(cache_key)
            if cached_embedding:
                all_embeddings.append(cached_embedding)
            else:
                uncached_chunks.append(chunk)
                uncached_indices.append(i)
    else:
        uncached_chunks = chunks
        uncached_indices = list(range(len(chunks)))

    # Обрабатываем некэшированные чанки пакетами
    for i in range(0, len(uncached_chunks), batch_size):
        batch = uncached_chunks[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        batch_embeddings = [data.embedding for data in response.data]
        
        # Сохраняем новые эмбеддинги в кэш (если кэширование включено) и добавляем их в результат
        for j, embedding in enumerate(batch_embeddings):
            if use_cache:
                chunk = uncached_chunks[i+j]
                cache_key = get_cache_key(chunk)
                save_to_cache(cache_key, embedding)
            all_embeddings.insert(uncached_indices[i+j], embedding)

    return all_embeddings

def save_chunks_and_embeddings(chunks: List[str], embeddings: List[List[float]], file_path: str):
    logger.info(f"Saving {len(chunks)} chunks and their embeddings to {file_path}.")
    data = {'chunks': chunks, 'embeddings': embeddings}
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Chunks and embeddings saved to {file_path}.")

def load_chunks_and_embeddings(file_path: str) -> Tuple[List[str], List[List[float]]]:
    logger.info(f"Loading chunks and embeddings from {file_path}.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data['chunks'])} chunks and embeddings.")
    return data['chunks'], data['embeddings']

def get_or_create_chunks_and_embeddings(text: Union[Dict[str, Any], List[str], str], cache_file: str, use_cache: bool = USE_CACHING) -> Tuple[List[str], List[List[float]]]:
    if use_cache and os.path.exists(cache_file) and not UPDATE_EMBEDDINGS:
        with open(cache_file, 'rb') as f:
            chunks, embeddings = pickle.load(f)
        if not all(isinstance(emb, list) and all(isinstance(x, float) for x in emb) for emb in embeddings):
            logger.warning("Cached embeddings are not in the correct format. Recreating embeddings.")
        else:
            return chunks, embeddings
    
    chunks = split_into_chunks(text)
    embeddings = create_embeddings(chunks, use_cache=use_cache)
    
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((chunks, embeddings), f)
    
    return chunks, embeddings

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_or_create_query_embedding(query: str, cache_file: str, use_cache: bool = USE_CACHING) -> List[float]:
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        if query in cache:
            return cache[query]
    
    embedding = create_embeddings([query], use_cache=use_cache)[0]
    
    if use_cache:
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
        
        cache[query] = embedding
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    
    return embedding
