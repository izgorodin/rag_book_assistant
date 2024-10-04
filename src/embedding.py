from typing import List, Tuple
import os
import pickle
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str]) -> List[List[float]]:
    logger.debug("Creating embeddings for chunks.")
    embeddings = []
    for idx, chunk in enumerate(chunks):
        try:
            logger.debug(f"Creating embedding for chunk {idx + 1}/{len(chunks)}.")
            response = client.embeddings.create(
                input=[chunk],
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating embedding for chunk {idx}: {str(e)}")
            raise
    logger.debug("Embeddings creation completed.")
    return embeddings

def save_chunks_and_embeddings(chunks: List[str], embeddings: List[List[float]], file_path: str):
    """Save both chunks and their embeddings to a single file."""
    logger.debug(f"Saving {len(chunks)} chunks and their embeddings to {file_path}.")
    data = {
        'chunks': chunks,
        'embeddings': embeddings
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Chunks and embeddings saved to {file_path}.")

def load_chunks_and_embeddings(file_path: str) -> Tuple[List[str], List[List[float]]]:
    """Load chunks and embeddings from a file."""
    logger.debug(f"Loading chunks and embeddings from {file_path}.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Chunks and embeddings loaded from {file_path}.")
    return data['chunks'], data['embeddings']

def get_or_create_chunks_and_embeddings(chunks: List[str], file_path: str) -> Tuple[List[str], List[List[float]]]:
    """Retrieve existing chunks and embeddings or create them if they don't exist."""
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