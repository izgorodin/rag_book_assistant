"""
Configuration Module

This module manages all configuration settings for the RAG system, including:
- API keys and credentials
- Model parameters
- System settings
- File paths and directories
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from src.error_handler import ConfigurationError
from src.types import Config

# Load environment variables
load_dotenv()

def get_required_env(key: str) -> str:
    """
    Get required environment variable or raise ConfigurationError.
    
    Args:
        key (str): Environment variable name
        
    Returns:
        str: Environment variable value
        
    Raises:
        ConfigurationError: If environment variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ConfigurationError(f"Required environment variable {key} is not set")
    return value

# OpenAI Configuration
OPENAI_CONFIG: Config = {
    'api_key': get_required_env("OPENAI_API_KEY"),
    'embedding_model': "text-embedding-3-small",
    'gpt_model': "gpt-4-turbo-preview",
    'max_tokens': 15000
}

# Pinecone Configuration
PINECONE_CONFIG: Config = {
    'api_key': get_required_env("PINECONE_API_KEY"),
    'environment': "us-east1-gcp",
    'dimension': 1536,
    'index_prefix': "book-",  # Префикс для индексов книг
    'cloud': "aws",
    'region': "us-east-1",
    'metric': "cosine"
}

# Text Processing Configuration
TEXT_PROCESSING_CONFIG: Config = {
    'chunk_size': 300,
    'overlap': 150,
    'top_k_chunks': 15
}

# File Paths Configuration
PATH_CONFIG: Config = {
    'book_path': 'tests/data/book.txt',
    'ford_path': 'tests/data/ford.txt',
    'embeddings_dir': 'data/embeddings',
    'cache_dir': 'data/cache'
}

def validate_config() -> None:
    """
    Validate all configuration settings.
    
    Raises:
        ConfigurationError: If any configuration is invalid
    """
    required_configs = [OPENAI_CONFIG, PINECONE_CONFIG, TEXT_PROCESSING_CONFIG, PATH_CONFIG]
    for config in required_configs:
        if not isinstance(config, dict):
            raise ConfigurationError(f"Invalid configuration type: {type(config)}")
