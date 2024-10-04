from typing import List
import os
import pickle
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=[chunk],
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    return embeddings

def save_embeddings(embeddings: List[List[float]], file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Embeddings saved to {file_path}")

def load_embeddings(file_path: str) -> List[List[float]]:
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    logger.info(f"Embeddings loaded from {file_path}")
    return embeddings

def get_or_create_embeddings(chunks: List[str], file_path: str) -> List[List[float]]:
    if os.path.exists(file_path):
        logger.info("Loading existing embeddings")
        return load_embeddings(file_path)
    else:
        logger.info("Creating new embeddings")
        embeddings = create_embeddings(chunks)
        save_embeddings(embeddings, file_path)
        return embeddings