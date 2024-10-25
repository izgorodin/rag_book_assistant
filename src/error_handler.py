from typing import Callable, Any
from functools import wraps
from src.logger import setup_logger

logger = setup_logger()

class RAGError(Exception):
    """Base exception class for RAG-related errors."""
    pass

class DataSourceError(RAGError):
    """Raised when there's an issue with the data source."""
    pass

class EmbeddingError(RAGError):
    """Raised when there's an issue with embeddings."""
    pass

class SearchError(RAGError):
    """Base class for search-related errors."""
    pass

class SimpleSearchError(SearchError):
    """Raised when there's an issue with simple search."""
    pass

class HybridSearchError(SearchError):
    """Raised when there's an issue with hybrid search."""
    pass

class QueryExpansionError(RAGError):
    """Raised when there's an issue with query expansion."""
    pass

class ScoreComputationError(RAGError):
    """Raised when there's an issue with score computation."""
    pass

class ConfigurationError(RAGError):
    """Raised when there's an issue with system configuration."""
    pass

class ModelError(RAGError):
    """Raised when there's an issue with the underlying model."""
    pass

class TokenizationError(RAGError):
    """Raised when there's an issue with text tokenization."""
    pass

class DimensionMismatchError(RAGError):
    """Raised when there's a mismatch in vector dimensions."""
    pass

class OpenAIError(RAGError):
    """Raised when there's an issue with OpenAI."""
    pass

def handle_rag_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except RAGError as e:
            logger.error(f"RAG Error in {func.__name__}: {str(e)}", exc_info=True)
            return str(e)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an unexpected error while processing your request: {str(e)}"
    return wrapper

def format_error_message(error: Exception) -> str:
    if isinstance(error, RAGError):
        return str(error)
    return f"Sorry, I encountered an unexpected error while processing your request: {str(error)}"
