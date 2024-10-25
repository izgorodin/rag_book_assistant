from src.cli import run_cli
import argparse
import logging

def main():
    # Initialize the logger
    logger = logging.getLogger('main.log')
    logger.info("Starting the RAG Book Assistant")

    parser = argparse.ArgumentParser(description="RAG Book Assistant")
    parser.add_argument("mode", choices=["cli"], help="Mode to run the assistant (only cli is currently supported)")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()  # Use the run_cli from cli.py
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
