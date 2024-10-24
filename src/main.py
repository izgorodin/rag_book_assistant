from src.logger import setup_logger
import argparse
import os
from src.text_processing import split_into_chunks
from src.rag import rag_query
import nltk
from src.embedding import get_or_create_chunks_and_embeddings
from src.file_processor import FileProcessor
from src.cli import run_cli  # Import run_cli from cli.py

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

def main():
    parser = argparse.ArgumentParser(description="RAG Book Assistant")
    parser.add_argument("mode", choices=["cli"], help="Mode to run the assistant (only cli is currently supported)")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()  # Use the run_cli from cli.py
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
