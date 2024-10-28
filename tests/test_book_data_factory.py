import pytest
from unittest.mock import Mock, MagicMock
from src.book_data_factory import BookDataFactory
from src.embedding import EmbeddingService
from src.vector_store_service import VectorStoreService

@pytest.fixture
def mock_services():
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.create_embeddings.return_value = [[0.1] * 1536]
    vector_store_service = MagicMock(spec=VectorStoreService)
    return embedding_service, vector_store_service

@pytest.fixture
def factory(mock_services):
    embedding_service, vector_store_service = mock_services
    return BookDataFactory(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service
    )

def test_create_from_text_success(factory):
    result = factory.create_from_text("Test content")
    assert result is not None
    assert len(result.get_chunks()) > 0
    assert len(result.get_embeddings()) > 0

def test_create_from_text_with_progress(mock_services):
    embedding_service, vector_store_service = mock_services
    progress_calls = []
    
    def progress_callback(status, current, total):
        progress_calls.append((status, current, total))
    
    factory = BookDataFactory(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        progress_callback=progress_callback
    )
    
    factory.create_from_text("Test content")
    assert len(progress_calls) > 0
    assert progress_calls[-1][1] == progress_calls[-1][2]  # Last call should complete

def test_retry_mechanism(mock_services):
    embedding_service, vector_store_service = mock_services
    embedding_service.create_embeddings.side_effect = [
        Exception("First attempt failed"),
        [[0.1] * 1536]  # Second attempt succeeds
    ]
    
    factory = BookDataFactory(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service
    )
    
    result = factory.create_from_text("Test content")
    assert result is not None
    assert embedding_service.create_embeddings.call_count == 2