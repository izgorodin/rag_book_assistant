from typing import List, Tuple, Dict, Any
import os
import pickle
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
import logging
import numpy as np
from src.text_processing import split_into_chunks
from src.cache import get_cache_key, save_to_cache, load_from_cache

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str]) -> List[List[float]]:
    all_embeddings = []
    for chunk in chunks:
        cache_key = get_cache_key(chunk)
        cached_embedding = load_from_cache(cache_key)
        if cached_embedding:
            all_embeddings.append(cached_embedding)
        else:
            response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embedding = response.data[0].embedding
            save_to_cache(cache_key, embedding)
            all_embeddings.append(embedding)
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

def get_or_create_chunks_and_embeddings(text: Dict[str, Any], file_path: str) -> Tuple[List[str], List[List[float]]]:
    chunks = split_into_chunks(text)
    if os.path.exists(file_path):
        logger.info("Loading existing chunks and embeddings.")
        loaded_chunks, embeddings = load_chunks_and_embeddings(file_path)
        if loaded_chunks != chunks:
            logger.warning("Chunks have changed. Recreating embeddings.")
            embeddings = create_embeddings(chunks)
            save_chunks_and_embeddings(chunks, embeddings, file_path)
    else:
        logger.info("Creating new chunks and embeddings.")
        embeddings = create_embeddings(chunks)
        save_chunks_and_embeddings(chunks, embeddings, file_path)
    return chunks, embeddings

def cosine_similarity(a: List[float], b: List[float]) -> float:
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
