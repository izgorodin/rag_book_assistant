from typing import Dict, Any, Optional, Callable, List, Union
from src.utils.logger import get_main_logger, get_rag_logger
from src.book_data_interface import BookDataInterface
from src.embedding import EmbeddingService
from src.services.text_processor import TextProcessor
from src.vector_store_service import VectorStoreService
from tqdm import tqdm
import time
import asyncio

logger = get_main_logger()
rag_logger = get_rag_logger()

class BookDataFactory:
    def __init__(self, 
                 embedding_service: EmbeddingService, 
                 vector_store_service: VectorStoreService,
                 progress_callback: Optional[Callable[[str, int, int], None]] = None):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.progress_callback = progress_callback
        self.logger = get_main_logger()
        self.text_processor = TextProcessor()

    def report_progress(self, status: str, current: int, total: int):
        """Синхронная версия report_progress"""
        if self.progress_callback:
            self.progress_callback(status, current, total)

    def process_text(self, text: str) -> tuple:
        """Process text into chunks."""
        preprocessed_text = self.text_processor.preprocess_text(text)
        chunks = self.text_processor.split_into_chunks(preprocessed_text)
        return chunks, [], [], []  # Возвращаем пустые списки вместо метаданных

    async def create_from_text(self, text: str) -> BookDataInterface:
        """Create BookDataInterface instance from text."""
        try:
            self.logger.info(f"Starting create_from_text with input type: {type(text)}")
            
            # Process text
            chunks, _, _, _ = self.process_text(text)
            self.report_progress("Text processing completed", 1, 4)
            
            # Generate embeddings
            embeddings = self.embedding_service.create_embeddings(chunks)
            self.report_progress("Embeddings generated", 2, 4)
            
            # Store vectors
            self.vector_store_service.store_vectors(chunks, embeddings)
            self.report_progress("Vectors stored", 3, 4)
            
            # Create and return interface
            book_data = BookDataInterface(
                chunks=chunks,
                embeddings=embeddings,
                processed_text={"original_text": text},
                embedding_service=self.embedding_service,
                vector_store_service=self.vector_store_service
            )
            self.report_progress("Book data interface created", 4, 4)
            
            return book_data
            
        except Exception as e:
            self.logger.error(f"Error in create_from_text: {str(e)}", exc_info=True)
            raise

    def _create_embeddings_with_retry(self, chunks: List[str], max_retries: int = 3) -> List[List[float]]:
        """Helper method to create embeddings with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.embedding_service.create_embeddings(chunks)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(2 ** attempt)
