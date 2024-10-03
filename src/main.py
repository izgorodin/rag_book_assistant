from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query

def main():
    # Load and preprocess the book
    book_path = input("Enter the path to the book file: ")
    text = load_and_preprocess_text(book_path)
    
    # Split into chunks and create embeddings
    chunks = split_into_chunks(text)
    embeddings = create_embeddings(chunks)
    
    print("Book successfully loaded and processed. You can ask questions!")
    
    # Question and answer loop
    while True:
        query = input("\nAsk a question about the book (or 'exit' to finish): ")
        if query.lower() == 'exit':
            break
        
        answer = rag_query(query, chunks, embeddings)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()