from typing import List, Tuple, Dict, Any
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from src.config import EMBEDDING_DIMENSION, OPENAI_API_KEY, EMBEDDING_MODEL
from src.logger import setup_logger
from src.text_processing import extract_dates, extract_named_entities, extract_key_phrases
from src.book_data_interface import BookDataInterface
from src.pinecone_manager import PineconeManager
from src.cache_manager import CACHE_DIR, save_to_cache, load_from_cache
import hashlib
from src.types import Chunk, EmbeddingList
from src.openai_service import OpenAIService
from src.error_handler import handle_rag_error, DataSourceError

logger = setup_logger()

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_manager = PineconeManager()
if pinecone_manager.index is None:
    print("Warning: Pinecone index is not available. Some functionality may be limited.")

@handle_rag_error
def create_embeddings(chunks: List[Chunk]) -> EmbeddingList:
    logger.info(f"Starting to create embeddings for {len(chunks)} chunks")
    openai_service = OpenAIService()
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = openai_service.create_embedding(chunk)
            embeddings.append(embedding)
            if (i + 1) % 100 == 0:
                logger.info(f"Created embeddings for {i + 1} chunks")
        except Exception as e:
            logger.error(f"Error creating embedding for chunk {i}: {str(e)}")
            raise DataSourceError(f"Error creating embedding for chunk {i}: {str(e)}")
    logger.info(f"Finished creating embeddings for {len(chunks)} chunks")
    return embeddings

def save_chunks_and_embeddings(book_data: BookDataInterface, file_path: str):
    logger.info(f"Saving {len(book_data.get_chunks())} chunks, their embeddings, and processed text to {file_path}.")
    with open(file_path, 'wb') as f:
        pickle.dump(book_data, f)
    logger.info(f"Data saved to {file_path}.")

def load_chunks_and_embeddings(file_path: str) -> Tuple[List[str], List[List[float]], Dict[str, Any]]:
    logger.info(f"Loading chunks, embeddings, and processed text from {file_path}.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data['chunks'])} chunks, embeddings, and processed text.")
    return data['chunks'], data.get('embeddings', []), data.get('processed_text', {})

@handle_rag_error
def get_or_create_chunks_and_embeddings(chunks: List[Chunk], file_path: str, index) -> BookDataInterface:
    if os.path.exists(file_path):
        logger.info(f"Loading existing chunks and embeddings from {file_path}")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded data type: {type(data)}")
            if isinstance(data, dict):
                logger.info("Creating BookDataInterface from dictionary")
                return BookDataInterface(data['chunks'], data['embeddings'], data.get('processed_text', {}))
            elif isinstance(data, BookDataInterface):
                logger.info("Loaded BookDataInterface object directly")
                return data
            else:
                logger.error(f"Unexpected data type in file: {type(data)}")
                raise ValueError(f"Unexpected data type in file: {type(data)}")
        except Exception as e:
            logger.error(f"Error loading from file: {str(e)}")
            raise DataSourceError(f"Error loading from file {file_path}: {str(e)}")
    else:
        logger.info("Creating new embeddings")
        try:
            embeddings = create_embeddings(chunks)
            logger.info(f"Created {len(embeddings)} embeddings")
            book_data = BookDataInterface(chunks, embeddings, {})
            book_data.save(file_path)
            logger.info(f"Saved book data to {file_path}")
            return book_data
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise DataSourceError(f"Error creating embeddings: {str(e)}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Добавляем проверку размерности
    if a.shape != b.shape:
        raise ValueError(f"Vectors have different shapes: {a.shape} and {b.shape}")
    
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

def load_from_cache(key: str) -> any:
    """
    Load data from the cache using a hashed key.

    Args:
        key (str): The cache key.

    Returns:
        any: The cached data if found, None otherwise.
    """
    hashed_key = hashlib.md5(key.encode()).hexdigest()
    file_path = os.path.join(CACHE_DIR, f"{hashed_key}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None
