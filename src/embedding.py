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
from src.pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_manager = PineconeManager()
if pinecone_manager.index is None:
    print("Warning: Pinecone index is not available. Some functionality may be limited.")

def create_embeddings(chunks: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Create embeddings for chunks in batches using OpenAI and store them in Pinecone.
    
    Args:
        chunks (List[str]): List of text chunks to embed.
        batch_size (int): Number of chunks to process in each batch.
    
    Returns:
        List[List[float]]: List of embeddings for all chunks.
    """
    def embedding_function(batch):
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]

    all_embeddings = pinecone_manager.get_or_create_embeddings(chunks, embedding_function)
    pinecone_manager.upsert_embeddings(chunks, all_embeddings)
    
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
