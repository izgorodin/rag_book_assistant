import os
from openai import OpenAI
from src.logger import setup_logger
from src.text_processing import load_and_preprocess_text
from src.embedding import EmbeddingService, create_book_data
from src.rag import rag_query
from src.book_data_interface import BookDataInterface
from src.openai_service import OpenAIService
from src.pinecone_manager import PineconeManager
from src.cache_manager import FileSystemCache, CacheManager
from src.config import OPENAI_API_KEY, CACHE_DIR

logger = setup_logger()

class BookAssistant:
    """Main class for handling book processing and question answering."""
    
    def __init__(self):
        """Initialize all necessary services."""
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.openai_service = OpenAIService()
        self.vector_store = PineconeManager()
        self.cache_manager = CacheManager(FileSystemCache(CACHE_DIR))
        
        self.embedding_service = EmbeddingService(
            openai_client=self.openai_client,
            vector_store=self.vector_store,
            cache_manager=self.cache_manager
        )
        logger.info("Book Assistant initialized")

    def load_and_process_book(self, book_path: str) -> BookDataInterface:
        """Load and process book from file."""
        try:
            text = load_and_preprocess_text(book_path)
            book_data = create_book_data(text, self.embedding_service)
            logger.info(f"Book data created. Number of chunks: {len(book_data.get_chunks())}")
            return book_data
        except Exception as e:
            logger.error(f"Error processing book: {str(e)}")
            raise

    def answer_question(self, query: str, book_data: BookDataInterface) -> str:
        """Generate answer for a question about the book."""
        logger.info(f"Processing query: {query}")
        try:
            answer = rag_query(query, book_data, self.openai_service)
            logger.info(f"Generated answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}"

    def run(self):
        """Run the interactive CLI session."""
        logger.info("Starting CLI session")
        
        try:
            # Load and process book
            while True:
                book_path = input("Enter the path to the book file: ").strip()
                if not book_path:
                    print("Please enter a valid path")
                    continue
                    
                if not os.path.exists(book_path):
                    print(f"File not found: {book_path}")
                    continue
                    
                break

            book_data = self.load_and_process_book(book_path)
            print("\nBook successfully loaded and processed. You can start asking questions!")
            
            # Question-answer loop
            while True:
                query = input("\nAsk a question (or type 'exit' to quit): ").strip()
                if query.lower() == 'exit':
                    break
                if not query:
                    continue
                    
                answer = self.answer_question(query, book_data)
                print(f"\nAnswer: {answer}")

        except Exception as e:
            logger.error(f"Session error: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}")
        
        logger.info("CLI session ended")
