from openai import OpenAI, AsyncOpenAI
from src.cli import BookAssistant
from src.cache_manager import CacheManager
from src.pinecone_manager import PineconeManager
from src.utils.logger import get_main_logger
from src.config import CACHE_DIR, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
import sys
import os
import asyncio
from src.vector_store_service import VectorStoreService
from src.embedding import EmbeddingService
from src.book_data_factory import BookDataFactory

logger = get_main_logger()

async def async_main():
    """Async main entry point for the application."""
    try:
        # Initialize cache manager
        cache_manager = CacheManager(cache_dir=CACHE_DIR)
        logger.info(f"CacheManager initialized at {CACHE_DIR}")

        # Initialize OpenAI client and services
        openai_client = AsyncOpenAI()
        
        # Initialize Pinecone store
        pinecone_store = PineconeManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX_NAME
        )
        
        # Create vector store service
        vector_store_service = VectorStoreService(pinecone_store)
        await vector_store_service.initialize()
        
        # Create embedding service
        embedding_service = EmbeddingService(
            openai_client=openai_client,
            cache_manager=cache_manager
        )
        
        # Create book data factory
        book_data_factory = BookDataFactory(
            embedding_service=embedding_service,
            vector_store_service=vector_store_service
        )
        
        # Initialize BookAssistant with all required services
        assistant = BookAssistant(
            openai_client=openai_client,
            cache_manager=cache_manager,
            vector_store_service=vector_store_service,
            book_data_factory=book_data_factory
        )
        
        await assistant.run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(async_main())