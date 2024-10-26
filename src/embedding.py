from typing import List, Tuple, Dict, Any
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from src.config import OPENAI_CONFIG
from src.logger import setup_logger
from src.book_data_interface import BookDataInterface
from src.cache_manager import CACHE_DIR, save_to_cache, load_from_cache
import hashlib
from src.types import APIKey, Chunk, EmbeddingList, Embedding
from src.openai_service import OpenAIService
from src.error_handler import handle_rag_error, DataSourceError

logger = setup_logger()
client = OpenAI(api_key=APIKey(OPENAI_CONFIG['api_key']))

@handle_rag_error
def create_embeddings(chunks: List[Chunk], file_path: str, batch_size: int = 5) -> None:
    logger.info(f"Starting to create embeddings for {len(chunks)} chunks")
    
    openai_service = OpenAIService(api_key=APIKey(OPENAI_CONFIG['api_key']))
    temp_file = f"{file_path}.temp"
    
    try:
        # Открываем файл один раз
        with open(temp_file, 'wb') as f:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = [openai_service.create_embedding(chunk) for chunk in batch]
                
                # Сразу сохраняем батч и очищаем память
                pickle.dump((batch, batch_embeddings), f)
                
                logger.info(f"Progress: Saved batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                del batch_embeddings
                del batch
        
        # Переименовываем временный файл
        os.rename(temp_file, file_path)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise DataSourceError(f"Failed to create embeddings: {str(e)}")
    
    logger.info("Successfully completed embedding creation and saving")

def save_chunks_and_embeddings(book_data: BookDataInterface, file_path: str) -> None:
    """
    Save book data (chunks, embeddings, and processed text) to a file.

    Args:
        book_data (BookDataInterface): Book data to be saved.
        file_path (str): Path to save the data.
    """
    logger.info(f"Saving {len(book_data.get_chunks())} chunks, their embeddings, and processed text to {file_path}.")
    with open(file_path, 'wb') as f:
        pickle.dump(book_data, f)
    logger.info(f"Data saved to {file_path}.")

def load_chunks_and_embeddings(file_path: str) -> Tuple[List[Chunk], EmbeddingList, Dict[str, Any]]:
    """
    Load book data (chunks, embeddings, and processed text) from a file.

    Args:
        file_path (str): Path to load the data from.

    Returns:
        Tuple[List[Chunk], EmbeddingList, Dict[str, Any]]: Loaded chunks, embeddings, and processed text.
    """
    logger.info(f"Loading chunks, embeddings, and processed text from {file_path}.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data['chunks'])} chunks, embeddings, and processed text.")
    return data['chunks'], EmbeddingList(data.get('embeddings', [])), data.get('processed_text', {})

@handle_rag_error
def get_or_create_chunks_and_embeddings(chunks: List[Chunk], file_path: str, batch_size: int = 5) -> BookDataInterface:
    if os.path.exists(file_path):
        logger.info(f"Loading existing data from {file_path}")
        return BookDataInterface.from_file(file_path)
    
    # Create embeddings and save them incrementally
    create_embeddings(chunks, file_path, batch_size)
    
    # Load the saved data
    return BookDataInterface.from_file(file_path)

def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """
    Calculate the cosine similarity between two embedding vectors.

    Args:
        a (Embedding): First embedding vector.
        b (Embedding): Second embedding vector.

    Returns:
        float: Cosine similarity between the two vectors.

    Raises:
        ValueError: If the vectors have different shapes.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    if a.shape != b.shape:
        raise ValueError(f"Vectors have different shapes: {a.shape} and {b.shape}")
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_or_create_query_embedding(query: str, cache_file: str) -> Embedding:
    """
    Get an existing query embedding from cache or create a new one.

    Args:
        query (str): The query text.
        cache_file (str): Path to the cache file.

    Returns:
        Embedding: The embedding for the query.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        if query in cache:
            return Embedding(cache[query])
    
    embedding = create_embeddings([Chunk(query)])[0]
    
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    
    cache[query] = embedding
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    return Embedding(embedding)

def load_from_cache(key: str) -> Any:
    """
    Load data from the cache using a hashed key.

    Args:
        key (str): The cache key.

    Returns:
        Any: The cached data if found, None otherwise.
    """
    hashed_key = hashlib.md5(key.encode()).hexdigest()
    file_path = os.path.join(CACHE_DIR, f"{hashed_key}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None
