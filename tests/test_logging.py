import pytest
from src.utils.logger import get_main_logger, get_rag_logger

def test_logger_initialization():
    """Test that loggers are properly initialized."""
    main_logger = get_main_logger()
    rag_logger = get_rag_logger()
    
    assert main_logger is not None
    assert rag_logger is not None
    assert main_logger.name == "main"
    assert rag_logger.name == "rag"
