from src.utils.logger import get_main_logger, get_rag_logger  # Import logging utilities for main and RAG logging
import argparse  # Import argparse for command-line argument parsing
import uvicorn  # Import uvicorn for running FastAPI applications
from src.cli import BookAssistant  # Import the BookAssistant class for CLI mode

logger = get_main_logger()  # Initialize the main logger
rag_logger = get_rag_logger()  # Initialize the RAG logger

def main():
    logger.info("Starting the RAG Book Assistant")  # Log the start of the application
    rag_logger.info("\nApplication Start\n" + "="*50)  # Log the application start in RAG logger
    
    parser = argparse.ArgumentParser(description="RAG Book Assistant")  # Create an argument parser
    parser.add_argument("mode", choices=["cli", "web", "api"],  # Define the mode argument with choices
                       help="Mode to run the assistant (cli, web, or api)")  # Help description for the mode argument
    args = parser.parse_args()  # Parse the command-line arguments

    try:
        if args.mode == "cli":  # Check if the mode is CLI
            assistant = BookAssistant()  # Initialize the BookAssistant
            assistant.run()  # Run the assistant
        elif args.mode == "web" or args.mode == "api":  # Check if the mode is web or API
            # Run FastAPI application
            uvicorn.run(
                "src.web.app:app",
                host="0.0.0.0",
                port=8080,
                reload=True,
                workers=1
            )
    except Exception as e:  # Catch any exceptions
        error_msg = f"Application error: {str(e)}"  # Create an error message
        logger.error(error_msg, exc_info=True)  # Log the error with traceback
        rag_logger.error(f"\nCritical Error:\n{error_msg}\n{'='*50}")  # Log critical error in RAG logger

if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Call the main function

"""
How to run:

# Run CLI mode
python -m src.main cli

# Run web/api mode
python -m src.main web

# Run with uvicorn directly (production)
uvicorn src.web.app:app --host 0.0.0.0 --port 8080 --workers 4
"""