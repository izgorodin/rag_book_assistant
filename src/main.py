import logging
import argparse
import os
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.rag import rag_query
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
from src.embedding import get_or_create_chunks_and_embeddings

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )

def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

def run_cli():
    setup_logging()
    initialize_nltk()
    logger = logging.getLogger(__name__)
    logger.info("Starting the RAG system")

    try:
        book_path = input("Enter the path to the book file: ")
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"The file {book_path} does not exist.")
        
        logger.info(f"Loading book from: {book_path}")
        text = load_and_preprocess_text(book_path)
        logger.info(f"Book loaded and preprocessed. Total length: {len(text)} characters")
        
        logger.info("Splitting text into chunks")
        chunks = split_into_chunks(text['text'])
        logger.info(f"Text split into {len(chunks)} chunks")
        
        chunks, embeddings = get_or_create_chunks_and_embeddings(chunks, 'embeddings_cache.pkl')
        
        print("Book successfully loaded and processed. You can ask questions!")
        
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
    parser.add_argument("mode", choices=["cli"], help="Mode to run the assistant (only cli is currently supported)")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
