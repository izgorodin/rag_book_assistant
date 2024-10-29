from typing import Dict, Any, Optional, Callable, List, Union
from src.utils.logger import get_main_logger, get_rag_logger
from src.book_data_interface import BookDataInterface
from src.embedding import EmbeddingService
from src.text_processing import load_and_preprocess_text, extract_dates, extract_named_entities, extract_key_phrases
from src.vector_store_service import VectorStoreService
from tqdm import tqdm
import time

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

    def create_from_text(self, input_data: Union[str, Dict[str, Any]]) -> BookDataInterface:
        """Create BookDataInterface from raw text or preprocessed data."""
        try:
            logger.info(f"Starting create_from_text with input type: {type(input_data)}")
            rag_logger.info(
                f"\nBook Processing Start:\n"
                f"Input type: {type(input_data)}\n"
                f"{'-'*50}"
            )
            
            if self.progress_callback:
                self.progress_callback("Starting text processing", 0, 4)
            
            # Get preprocessed data
            logger.info("Calling load_and_preprocess_text...")
            preprocessed_data = load_and_preprocess_text(input_data)
            logger.info(f"Preprocessed data type: {type(preprocessed_data)}")
            logger.info(f"Preprocessed data keys: {preprocessed_data.keys()}")
            
            chunks = preprocessed_data['chunks']
            logger.info(f"Chunks type: {type(chunks)}")
            logger.info(f"First chunk type (if exists): {type(chunks[0]) if chunks else 'no chunks'}")
            
            if not chunks:
                raise ValueError("No chunks found in preprocessed data")
            
            if self.progress_callback:
                self.progress_callback("Extracting features", 1, 4)

            # Convert chunks to text
            logger.info("Converting chunks to text...")
            text_chunks = [str(chunk) for chunk in chunks]
            logger.info(f"Text chunks type: {type(text_chunks)}")
            logger.info(f"First text chunk type: {type(text_chunks[0])}")
            
            # Extract features
            logger.info("Extracting features...")
            dates = extract_dates(text_chunks)
            logger.info(f"Dates extracted: {len(dates)}")
            entities = extract_named_entities(text_chunks)
            logger.info(f"Entities extracted: {len(entities)}")
            key_phrases = extract_key_phrases(text_chunks)
            logger.info(f"Key phrases extracted: {len(key_phrases)}")
            
            logger.info("Features extracted successfully")
            rag_logger.info(
                f"\nFeature Extraction:\n"
                f"Chunks: {len(chunks)}\n"
                f"Dates: {len(dates)}\n"
                f"Entities: {len(entities)}\n"
                f"Key phrases: {len(key_phrases)}\n"
                f"{'-'*50}"
            )
            
            if self.progress_callback:
                self.progress_callback("Creating embeddings", 2, 4)
            
            # Create embeddings
            logger.info("Creating embeddings...")
            embeddings = self._create_embeddings_with_retry(text_chunks)
            logger.info(f"Embeddings created: {len(embeddings)}")
            
            if self.progress_callback:
                self.progress_callback("Storing vectors", 3, 4)
            
            # Store vectors
            logger.info("Storing vectors...")
            self.vector_store_service.store_vectors(text_chunks, embeddings)
            logger.info("Vectors stored successfully")
            
            if self.progress_callback:
                self.progress_callback("Completed", 4, 4)
            
            logger.info("Creating BookDataInterface instance...")
            return BookDataInterface(
                chunks=text_chunks,
                embeddings=embeddings,
                processed_text=preprocessed_data,
                embedding_service=self.embedding_service,  # Pass the service
                dates=dates,
                entities=entities,
                key_phrases=key_phrases
            )
        except Exception as e:
            error_msg = f"Error in create_from_text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            rag_logger.error(f"\nBook Processing Error:\n{error_msg}\n{'-'*50}")
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
