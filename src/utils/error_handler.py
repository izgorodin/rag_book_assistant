from typing import Callable, Any, Dict, Optional
from functools import wraps
from src.utils.logger import get_main_logger, get_rag_logger
import sys

logger = get_main_logger()
rag_logger = get_rag_logger()

class RAGError(Exception):
    """Base exception class for RAG-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message

class FileProcessingError(RAGError):
    """Error raised during file processing."""
    def __init__(self, file_path: str, error_type: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Unsupported file format",
            details={
                'file_path': file_path,
                'error_type': error_type,
                **(details or {})
            }
        )

def handle_rag_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            rag_logger.error(f"\nFunction: {func.__name__}\nError: {str(e)}\n{'='*50}")
            
            # В тестовом окружении пробрасываем исключение дальше
            if 'pytest' in sys.modules:
                raise
                
            # В продакшене возвращаем отформатированное сообщение
            if isinstance(e, RAGError):
                return str(e)
            return format_error_message(e)
    return wrapper

def format_error_message(error: Exception) -> str:
    return f"Sorry, I encountered an error while processing your request: {str(error)}"

