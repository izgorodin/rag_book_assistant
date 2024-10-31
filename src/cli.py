import os  # Importing os module for file path operations
from openai import OpenAI  # Importing OpenAI client for API interactions
from src.book_data_factory import BookDataFactory  # Importing factory for creating book data
from src.services.file_processor import FileProcessor  # Importing file processor for handling book files
from src.utils.logger import get_main_logger, get_rag_logger  # Importing logging utilities
from src.embedding import EmbeddingService  # Importing embedding service for generating embeddings
from src.rag import rag_query  # Importing function for querying the RAG system
from src.book_data_interface import BookDataInterface  # Importing interface for book data handling
from src.openai_service import OpenAIService  # Importing OpenAI service for API interactions
from src.pinecone_manager import PineconeManager  # Importing Pinecone manager for vector storage
from src.cache_manager import CacheManager  # Importing cache manager for caching functionalities
from src.config import OPENAI_API_KEY, CACHE_DIR  # Importing configuration constants
from typing import Union, TextIO  # Importing types for type hinting
from src.vector_store_service import VectorStoreService  # Importing vector store service for managing embeddings
from tqdm import tqdm  # Importing tqdm for progress bar functionality
import sys  # Importing sys for system-specific parameters and functions
import click  # Importing click for CLI command-line parsing
from src.services.text_processor import print_chunks_analysis  # Importing text processor for analyzing chunks

logger = get_main_logger()  # Initializing the main logger
rag_logger = get_rag_logger()  # Initializing the RAG logger

def create_progress_bar(desc: str, total: int) -> tqdm:
    """Create a progress bar with consistent styling."""
    return tqdm(
        total=total,  # Total number of iterations for the progress bar
        desc=desc,  # Description to display on the progress bar
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',  # Format of the progress bar
        file=sys.stdout,  # Output to standard output
        ncols=80  # Width of the progress bar
    )

def progress_callback(progress_info):
    """Callback функция для отображения прогресса"""
    # Используем существующий create_progress_bar
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = create_progress_bar(
            progress_info['message'], 
            progress_info['total']
        )
    
    # Обновляем прогресс
    current = progress_info['current'] - progress_callback.pbar.n
    progress_callback.pbar.update(current)
    
    # Закрываем если завершено
    if progress_info['progress'] >= 100:
        progress_callback.pbar.close()
        delattr(progress_callback, 'pbar')

class BookAssistant:
    """Main class for handling book processing and question answering."""
    
    def __init__(self, progress_callback=progress_callback):
        """Initialize all necessary services."""
        # Initialize base services
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)  # Initialize OpenAI client with API key
        self.vector_store = PineconeManager(lazy_init=False)  # Initialize Pinecone manager
        self.cache_manager = CacheManager(CACHE_DIR)  # Initialize cache manager with cache directory
        
        # Initialize processing services
        self.openai_service = OpenAIService()  # Initialize OpenAI service
        self.embedding_service = EmbeddingService(
            openai_client=self.openai_client,  # Pass OpenAI client to embedding service
            cache_manager=self.cache_manager,  # Pass cache manager to embedding service
            progress_callback=self.update_progress  # Set progress callback for embedding service
        )
        self.vector_store_service = VectorStoreService(vector_store=self.vector_store)  # Initialize vector store service
        
        # Initialize factory
        self.book_data_factory = BookDataFactory(
            embedding_service=self.embedding_service,  # Pass embedding service to factory
            vector_store_service=self.vector_store_service,  # Pass vector store service to factory
            progress_callback=progress_callback  # Set progress callback for factory
        )
        logger.info("Book Assistant initialized")  # Log initialization of Book Assistant
        rag_logger.info("\nSystem Initialization:\nStatus: Ready\n" + "-"*50)  # Log system status

    def load_and_process_book(self, input_data: Union[str, TextIO]) -> BookDataInterface:
        """Load and process book from file path or text content."""
        try:
            # Get text content from input data
            if isinstance(input_data, str):
                if os.path.exists(input_data):  # Check if input is a valid file path
                    file_processor = FileProcessor()  # Initialize file processor
                    text = file_processor.process_file(input_data)  # Process the file to get text
                else:
                    text = input_data  # Use the input string as text
            else:
                text = input_data.read()  # Read text from TextIO object
            
            if not text:  # Check if text is empty
                raise ValueError("Empty text content")  # Raise error for empty content
                
            logger.info(f"Text content loaded, length: {len(text)}")  # Log length of loaded text
            rag_logger.info(
                f"\nBook Loading:\n"
                f"Content length: {len(text)} chars\n"
                f"{'-'*50}"
            )
            return self.book_data_factory.create_from_text(text)  # Create and return BookDataInterface from text
        except Exception as e:
            error_msg = f"Error processing book: {str(e)}"  # Prepare error message
            logger.error(error_msg)  # Log error
            rag_logger.error(f"\nProcessing Error:\n{error_msg}\n{'-'*50}")  # Log processing error
            raise  # Raise the exception

    def answer_question(self, query: str, book_data: BookDataInterface) -> str:
        """Generate answer for a question about the book."""
        logger.info(f"Processing query: {query}")  # Log the query being processed
        rag_logger.info(
            f"\nQuery Processing:\n"
            f"Query: {query}\n"
            f"{'-'*50}"
        )
        try:
            answer = rag_query(query, book_data, self.openai_service, self.embedding_service)  # Generate answer using RAG query
            logger.info("Answer generated successfully")  # Log successful answer generation
            rag_logger.info(
                f"\nAnswer Generated:\n"
                f"Length: {len(answer)} chars\n"
                f"{'-'*50}"
            )
            return answer  # Return the generated answer
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"  # Prepare error message
            logger.error(error_msg, exc_info=True)  # Log error with traceback
            rag_logger.error(f"\nAnswer Generation Error:\n{error_msg}\n{'-'*50}")  # Log answer generation error
            return f"An error occurred: {str(e)}"  # Return error message

    def run(self):
        """Run the interactive CLI session."""
        logger.info("Starting CLI session")  # Log the start of the CLI session
        
        try:
            # Load and process book
            while True:
                book_path = input("Enter the path to the book file: ").strip()  # Prompt for book file path
                if not book_path:  # Check if path is empty
                    print("Please enter a valid path")  # Prompt for valid path
                    continue
                    
                if not os.path.exists(book_path):  # Check if file exists
                    print(f"File not found: {book_path}")  # Notify user if file not found
                    continue
                    
                break  # Exit loop if valid path is provided

            book_data = self.load_and_process_book(book_path)  # Load and process the book
            print("\nBook successfully loaded and processed. You can start asking questions!")  # Notify user of successful loading
            
            # Question-answer loop
            while True:
                query = input("\nAsk a question (or type 'exit' to quit): ").strip()  # Prompt for user query
                if query.lower() == 'exit':  # Check for exit command
                    break  # Exit loop if user types 'exit'
                if not query:  # Check if query is empty
                    continue  # Skip to next iteration if empty
                    
                answer = self.answer_question(query, book_data)  # Generate answer for the query
                print(f"\nAnswer: {answer}")  # Display the answer

        except Exception as e:
            logger.error(f"Session error: {str(e)}", exc_info=True)  # Log session error with traceback
            print(f"An error occurred: {str(e)}")  # Notify user of the error
        
        logger.info("CLI session ended")  # Log the end of the CLI session

    def update_progress(self, desc: str, current: int, total: int, pbar=None):
        """Update progress bar with current status."""
        if not hasattr(self, '_pbar') or self._pbar is None:  # Check if progress bar is initialized
            self._pbar = create_progress_bar(desc, total)  # Create a new progress bar
        
        # Update description if it has changed
        if self._pbar.desc != desc:
            self._pbar.set_description(desc)  # Set new description for the progress bar
        
        # Update progress
        self._pbar.update(current - self._pbar.n)  # Update the progress bar with the current progress
        
        # Close if completed
        if current >= total:  # Check if progress is complete
            self._pbar.close()  # Close the progress bar
            self._pbar = None  # Reset progress bar attribute

@click.group()
def cli():
    """CLI для работы с RAG Book Assistant"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--chunk-size', default=1000, help='Size of each chunk')
@click.option('--overlap', default=150, help='Overlap between chunks')
def analyze_file(file_path, chunk_size, overlap):
    """Анализирует файл и показывает информацию о чанках"""
    try:
        processor = FileProcessor()
        text = processor.process_file(file_path)
        print_chunks_analysis(text, chunk_size, overlap)
    except Exception as e:
        click.echo(f"Error analyzing file: {str(e)}", err=True)

if __name__ == '__main__':
    cli()  # Теперь не нужно оборачивать в asyncio.run()
