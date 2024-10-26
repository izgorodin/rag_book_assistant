import os
import hashlib
import psutil
from typing import List

from openai import OpenAIError
from src.error_handler import DataSourceError, RAGError, handle_rag_error
from src.logger import setup_logger
from src.text_processing import load_and_preprocess_text
from src.embedding import get_or_create_chunks_and_embeddings
from src.rag import rag_query
from src.book_data_interface import BookDataInterface
from src.openai_service import OpenAIService
from src.types import APIKey, ProcessedBook, QueryType
from src.config import PATH_CONFIG, OPENAI_CONFIG
from src.pinecone_manager import PineconeManager

logger = setup_logger()

@handle_rag_error
def load_and_process_book(file_path: str) -> BookDataInterface:
    """
    Process book from file path instead of keeping content in memory
    """
    logger.info("=== Starting book processing pipeline ===")
    
    processed_text: ProcessedBook = load_and_preprocess_text(file_path)
    chunk_count = len(processed_text['chunks'])
    
    logger.info(f"Text preprocessed. Generated {chunk_count} chunks")
    
    book_data = get_or_create_chunks_and_embeddings(
        processed_text['chunks'], 
        file_path
    )
    
    return book_data

@handle_rag_error
def answer_question(query: QueryType, book_data: BookDataInterface, openai_service: OpenAIService) -> str:
    """
    Generate an answer to the given query using the RAG system.

    Args:
        query (QueryType): The user's question.
        book_data (BookDataInterface): The processed book data.
        openai_service (OpenAIService): The OpenAI service for generating answers.

    Returns:
        str: The generated answer.

    Raises:
        RAGError: If there's an error in the RAG process.
    """
    logger.info(f"Received query: {query}")
    return rag_query(query, book_data, openai_service)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def run_cli():
    """
    Run the Command Line Interface for the RAG Book Assistant.
    """
    logger.info("Starting the RAG system")

    try:
        openai_service = OpenAIService(api_key=APIKey(OPENAI_CONFIG['api_key']))

        book_path = input("Enter the path to the book file: ")
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"The file {book_path} does not exist.")
        
        # Создаем один экземпляр PineconeManager
        book_id = hashlib.md5(book_path.encode()).hexdigest()
        pinecone_manager = PineconeManager(project_id=book_id)
        
        # Передаем его в функции, которые его используют
        book_data = load_and_process_book(book_path)
        
        print("Book successfully loaded and processed. You can ask questions!")
        
        while True:
            query = input("\nAsk a question about the book (or 'exit' to finish): ")
            if query.lower() == 'exit':
                break
            
            answer = answer_question(QueryType(query), book_data, openai_service)
            print(f"\nAnswer: {answer}")

    except (DataSourceError, OpenAIError, RAGError) as e:
        logger.error(f"Error in RAG system: {str(e)}")
        print(f"An error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")

    logger.info("Exiting the RAG system")

if __name__ == "__main__":
    run_cli()
