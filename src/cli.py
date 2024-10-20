import logging
import argparse
import os
from tqdm import tqdm
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import get_or_create_chunks_and_embeddings
from src.rag import rag_query

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )

def load_and_process_book(book_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading book from: {book_path}")
    text = load_and_preprocess_text(book_path)
    logger.info("Book loaded and preprocessed")
    
    logger.info("Creating or loading embeddings")
    embeddings_file = os.path.join("data", "embeddings", f"{os.path.splitext(os.path.basename(book_path))[0]}_chunks_embeddings.pkl")
    chunks, embeddings = get_or_create_chunks_and_embeddings(text, embeddings_file)
    logger.info(f"Text split into {len(chunks)} chunks")
    logger.info("Embeddings created or loaded")
    
    return chunks, embeddings

def answer_question(query, chunks, embeddings):
    logger = logging.getLogger(__name__)
    logger.info(f"Received query: {query}")
    answer = rag_query(query, chunks, embeddings)
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
        text = load_and_preprocess_text(book_path)
        logger.info("Book loaded and preprocessed")
        
        # Create or load embeddings
        logger.info("Creating or loading embeddings")
        embeddings_file = os.path.join("data", "embeddings", f"{os.path.splitext(os.path.basename(book_path))[0]}_chunks_embeddings.pkl")
        chunks, embeddings = get_or_create_chunks_and_embeddings(text, embeddings_file)
        logger.info(f"Text split into {len(chunks)} chunks")
        logger.info("Embeddings created or loaded")
        
        print("Book successfully loaded and processed. You can ask questions!")
        
        # Question and answer loop
        while True:
            query = input("\nAsk a question about the book (or 'exit' to finish): ")
            if query.lower() == 'exit':
                break
            
            logger.info(f"Received query: {query}")
            answer = rag_query(query, chunks, embeddings)
            logger.info(f"Generated answer: {answer}")
            print(f"\nAnswer: {answer}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

    logger.info("Exiting the RAG system")

def main():
    parser = argparse.ArgumentParser(description="RAG Book Assistant")
    parser.add_argument("mode", choices=["cli", "api"], help="Mode to run the assistant (cli or api)")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "api":
        print("API mode not implemented yet.")
        # TODO: Implement API mode
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
