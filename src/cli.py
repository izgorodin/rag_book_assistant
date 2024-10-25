import os
import hashlib
from typing import List

from openai import OpenAIError
from src.error_handler import DataSourceError, RAGError, handle_rag_error
from src.logger import setup_logger
from src.text_processing import load_and_preprocess_text
from src.embedding import get_or_create_chunks_and_embeddings
from src.rag import rag_query
from src.book_data_interface import BookDataInterface
from src.openai_service import OpenAIService
from src.types import Chunk, QueryType

logger = setup_logger()

@handle_rag_error
def load_and_process_book(text_content: str) -> BookDataInterface:
    """
    Process the book content by preprocessing the text and creating embeddings.

    Args:
        text_content (str): The raw text content of the book.

    Returns:
        BookDataInterface: An object containing the processed book data.

    Raises:
        DataSourceError: If there's an issue processing the book data.
    """
    logger.info("Starting to process book content")
    
    processed_text = load_and_preprocess_text(text_content)
    logger.info(f"Text preprocessed. Number of chunks: {len(processed_text['chunks'])}")
    for i, chunk in enumerate(processed_text['chunks']):
        logger.info(f"Chunk {i+1} content ({len(chunk)} chars): {chunk[:100]}...")
    
    content_hash = hashlib.md5(text_content.encode()).hexdigest()
    
    embeddings_dir = os.path.join("data", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings_file = os.path.join(embeddings_dir, f"{content_hash}_chunks_embeddings.pkl")
    
    book_data = get_or_create_chunks_and_embeddings(processed_text['chunks'], embeddings_file)
    logger.info(f"Book data created. Type: {type(book_data)}")
    
    if isinstance(book_data, BookDataInterface):
        logger.info(f"Book data created. Number of chunks: {len(book_data.get_chunks())}")
        logger.info(f"First chunk content: {book_data.get_chunks()[0][:100]}...")
    else:
        logger.error(f"Unexpected book_data type: {type(book_data)}")
    
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

def run_cli():
    """
    Run the Command Line Interface for the RAG Book Assistant.
    Handles user input, book processing, and question-answering loop.
    """
    logger.info("Starting the RAG system")

    try:
        openai_service = OpenAIService()

        book_path = input("Enter the path to the book file: ")
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"The file {book_path} does not exist.")
        
        logger.info(f"Loading book from: {book_path}")
        with open(book_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        logger.info(f"File content loaded: {len(text_content)} characters")
        
        book_data = load_and_process_book(text_content)
        logger.info(f"Book loaded and preprocessed. Text split into {len(book_data.get_chunks())} chunks")
        
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
