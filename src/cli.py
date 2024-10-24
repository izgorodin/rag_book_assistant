import logging
import argparse
import os
import hashlib
from src.text_processing import load_and_preprocess_text
from src.embedding import get_or_create_chunks_and_embeddings
from src.rag import rag_query
from src.book_data_interface import BookDataInterface

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )

def load_and_process_book(text_content: str) -> BookDataInterface:
   
    logger.info("Starting to process book content")
    
    processed_text = load_and_preprocess_text(text_content)
    logger.info(f"Text preprocessed. Number of chunks: {len(processed_text['chunks'])}")
    
    # Generate a unique identifier for this text content
    content_hash = hashlib.md5(text_content.encode()).hexdigest()
    
    embeddings_dir = os.path.join("data", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings_file = os.path.join(embeddings_dir, f"{content_hash}_chunks_embeddings.pkl")
    
    book_data = get_or_create_chunks_and_embeddings(processed_text['chunks'], embeddings_file)
    logger.info(f"Book data created. Number of chunks: {len(book_data.chunks)}")
    
    return book_data

def answer_question(query: str, book_data: BookDataInterface) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f"Received query: {query}")
    answer = rag_query(query, book_data)
    logger.info(f"Generated answer: {answer}")
    return answer

def run_cli():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting the RAG system")

    try:
        # Load and preprocess the book
        book_path = input("Enter the path to the book file: ")
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"The file {book_path} does not exist.")
        
        logger.info(f"Loading book from: {book_path}")
        book_data = load_and_process_book(book_path)
        logger.info(f"Book loaded and preprocessed. Text split into {len(book_data.chunks)} chunks")
        
        print("Book successfully loaded and processed. You can ask questions!")
        
        # Question and answer loop
        while True:
            query = input("\nAsk a question about the book (or 'exit' to finish): ")
            if query.lower() == 'exit':
                break
            
            logger.info(f"Received query: {query}")
            try:
                answer = rag_query(query, book_data)
                logger.info(f"Generated answer: {answer}")
                print(f"\nAnswer: {answer}")
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                print(f"An error occurred while generating the answer: {str(e)}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

    logger.info("Exiting the RAG system")

def main():
    parser = argparse.ArgumentParser(description="RAG Book Assistant")
    parser.add_argument("mode", choices=["cli", "api"], help="Mode to run the assistant (cli or api)")
    args = parser.parse_args()

    try:
        if args.mode == "cli":
            run_cli()
        elif args.mode == "api":
            print("API mode not implemented yet.")
            # TODO: Implement API mode
        else:
            print(f"Unknown mode: {args.mode}")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
