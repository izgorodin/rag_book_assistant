import logging
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
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

def main():
    setup_logging()
    logging.info("Starting the RAG system")

    # Load and preprocess the book
    book_path = input("Enter the path to the book file: ")
    logging.info(f"Loading book from: {book_path}")
    text = load_and_preprocess_text(book_path)
    logging.info("Book loaded and preprocessed")
    
    # Split into chunks and create embeddings
    logging.info("Splitting text into chunks")
    chunks = split_into_chunks(text)
    logging.info(f"Text split into {len(chunks)} chunks")
    
    logging.info("Creating embeddings")
    embeddings = create_embeddings(chunks)
    logging.info("Embeddings created")
    
    print("Book successfully loaded and processed. You can ask questions!")
    
    # Question and answer loop
    while True:
        query = input("\nAsk a question about the book (or 'exit' to finish): ")
        if query.lower() == 'exit':
            break
        
        logging.info(f"Received query: {query}")
        answer = rag_query(query, chunks, embeddings)
        logging.info(f"Generated answer: {answer}")
        print(f"\nAnswer: {answer}")

    logging.info("Exiting the RAG system")

if __name__ == "__main__":
    main()