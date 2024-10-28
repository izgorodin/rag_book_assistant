from src.utils.logger import setup_logger
import argparse
import nltk
from src.cli import BookAssistant

logger = setup_logger('main.log')

def initialize_nltk():
    """Initialize NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

def main():
    logger.info("Starting the RAG Book Assistant")
    
    parser = argparse.ArgumentParser(description="RAG Book Assistant")
    parser.add_argument("mode", choices=["cli", "web", "api"], 
                       help="Mode to run the assistant (cli, web, or api)")
    args = parser.parse_args()

    initialize_nltk()

    try:
        if args.mode == "cli":
            assistant = BookAssistant()
            assistant.run()
        elif args.mode == "web":
            from src.web.app import run_web_app
            run_web_app()
        elif args.mode == "api":
            logger.info("API mode not implemented yet")
            print("API mode not implemented yet")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()


"""
# Run CLI mode
python -m src.main cli

# Run web mode
python -m src.main web

# Run with gunicorn (production)
gunicorn --bind 0.0.0.0:8080 --worker-class eventlet wsgi:app

"""