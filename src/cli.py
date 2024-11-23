import asyncio
import sys
from tqdm import tqdm

from openai import OpenAI
from src.cache_manager import CacheManager
from src.config import CACHE_DIR, OPENAI_API_KEY
from src.embedding import EmbeddingService
from src.pinecone_manager import PineconeManager
from src.services.llm_service import LLMService
from src.utils.logger import get_main_logger, get_rag_logger
from src.services.file_processor import FileProcessor
from src.book_data_factory import BookDataFactory
from src.book_data_interface import BookDataInterface
from src.openai_service import OpenAIService
from src.vector_store_service import VectorStoreService

class BookAssistant:
    """Main class for handling book processing and question answering."""
    
    def create_progress_bar(self, desc: str, total: int) -> tqdm:
        """Create a progress bar with consistent styling."""
        return tqdm(
            total=total,
            desc=desc,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            file=sys.stdout,
            ncols=80
        )

    def update_progress(self, desc: str, current: int, total: int, pbar=None):
        """Update progress bar with current status."""
        if not hasattr(self, '_pbar') or self._pbar is None:
            self._pbar = self.create_progress_bar(desc, total)
        
        if self._pbar.desc != desc:
            self._pbar.set_description(desc)
        
        self._pbar.update(current - self._pbar.n)
        
        if current >= total:
            self._pbar.close()
            self._pbar = None

    def __init__(self, openai_client, cache_manager):
        """Initialize the Book Assistant with necessary services."""
        self.logger = get_main_logger()
        
        # Инициализация сервисов
        vector_store = PineconeManager()
        self.vector_store_service = VectorStoreService(vector_store)
        self.embedding_service = EmbeddingService(openai_client, cache_manager)
        self.llm_service = LLMService(openai_client)
        self.file_processor = FileProcessor()
        
        # Создаем фабрику для обработки книг
        self.book_data_factory = BookDataFactory(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service
        )
        
        self.logger.info("Book Assistant initialized")

    async def load_and_process_book(self, book_path: str) -> BookDataInterface:
        """Load and process a book file"""
        try:
            text = self.file_processor.process_file(book_path)
            self.logger.info(f"Text content loaded, length: {len(text)}")
            return await self.book_data_factory.create_from_text(text)
        except Exception as e:
            self.logger.error(f"Error processing book: {str(e)}", exc_info=True)
            raise

    async def process_question(self, question: str, book_data: BookDataInterface) -> str:
        """Process a question using the loaded book data."""
        try:
            # Получаем релевантные чанки
            relevant_chunks = book_data.get_relevant_chunks(question)
            context = "\n".join(relevant_chunks)
            
            # Передаем и query и context в generate_response
            response = await self.llm_service.generate_response(
                query=question,  # Передаем вопрос как query
                context=context  # Передаем контекст
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def run(self):
        """Run the CLI session"""
        try:
            while True:
                book_path = input("Enter the path to the book file: ")
                if book_path.lower() == 'exit':
                    break
                
                loop = asyncio.get_event_loop()
                book_data = loop.run_until_complete(self.load_and_process_book(book_path))
                
                while True:
                    question = input("\nEnter your question (or 'exit' to change book): ")
                    if question.lower() == 'exit':
                        break
                    
                    answer = loop.run_until_complete(self.process_question(question, book_data))
                    print(f"\nAnswer: {answer}")
                    
        except Exception as e:
            self.logger.error(f"Session error: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}")
        finally:
            self.logger.info("CLI session ended")

def main():
    """Entry point for the CLI application"""
    try:
        assistant = BookAssistant()
        assistant.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
