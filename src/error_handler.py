from typing import Callable, Any
from functools import wraps
from src.logger import setup_logger

logger = setup_logger()

class RAGError(Exception):
    """Base exception class for RAG-related errors."""
    pass

def handle_rag_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            if isinstance(e, RAGError):
                return str(e)
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    return wrapper

def format_error_message(error: Exception) -> str:
    return f"Sorry, I encountered an error while processing your request: {str(error)}"

