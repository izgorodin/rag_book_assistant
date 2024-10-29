from typing import Callable, Any
from functools import wraps
from src.utils.logger import get_main_logger, get_rag_logger

logger = get_main_logger()
rag_logger = get_rag_logger()

class RAGError(Exception):
    """Base exception class for RAG-related errors."""
    pass

def handle_rag_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            rag_logger.error(f"\nFunction: {func.__name__}\nError: {str(e)}\n{'='*50}")
            if isinstance(e, RAGError):
                return str(e)
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    return wrapper

def format_error_message(error: Exception) -> str:
    return f"Sorry, I encountered an error while processing your request: {str(error)}"

