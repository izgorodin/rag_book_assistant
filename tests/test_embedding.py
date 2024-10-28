import pytest
from unittest.mock import Mock, patch, call
import numpy as np
from src.embedding import EmbeddingService, EmbeddingServiceError, cosine_similarity, create_book_data, EmbeddingError, EmbeddingDimensionError, RateLimitError, APIError, APITimeoutError, APIConnectionError
from src.book_data_interface import BookDataInterface
from src.utils.metrics import MetricsCollector

@pytest.fixture
def embedding_service():
    """Create EmbeddingService with mocked dependencies."""
    mock_openai = Mock()
    mock_cache_manager = Mock()
    mock_metrics = Mock(spec=MetricsCollector)
    return EmbeddingService(
        openai_client=mock_openai,
        cache_manager=mock_cache_manager,
        batch_size=2,
        metrics_collector=mock_metrics
    )

class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    def test_create_embeddings_batch_processing(self, embedding_service):
        """Test that chunks are processed in correct batch sizes."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        mock_embedding = [0.1] * 1536
        
        # Mock cache miss for all chunks
        embedding_service.cache_manager.load_async.return_value = None
        
        # Mock OpenAI response
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_embedding)] * 2,  # First batch
            response_ms=100
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Check batch processing
        assert embedding_service.client.embeddings.create.call_count == 2
        calls = embedding_service.client.embeddings.create.call_args_list
        assert calls[0][1]['input'] == ["chunk1", "chunk2"]
        assert calls[1][1]['input'] == ["chunk3"]

    def test_create_embeddings_caching(self, embedding_service):
        """Test caching behavior for embeddings."""
        chunks = ["cached", "uncached"]
        mock_cached = [0.1] * 1536
        mock_new = [0.2] * 1536
        
        # Mock cache hit for first chunk, miss for second
        embedding_service.cache_manager.load_async.side_effect = [mock_cached, None]
        
        # Mock OpenAI response
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_new)],
            response_ms=100
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        assert embeddings[0] == mock_cached
        assert embeddings[1] == mock_new
        assert embedding_service.client.embeddings.create.call_count == 1

    def test_create_embeddings_dimension_check(self, embedding_service):
        """Test dimension validation for embeddings."""
        chunks = ["test"]
        wrong_dim_embedding = [0.1] * 100  # Wrong dimension
        
        embedding_service.cache_manager.load_async.return_value = None
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=wrong_dim_embedding)],
            response_ms=100
        )
        
        with pytest.raises(EmbeddingDimensionError):
            embedding_service.create_embeddings(chunks)

    def test_error_handling(self, embedding_service):
        """Test handling of OpenAI API errors."""
        chunks = ["test"]
        embedding_service.cache_manager.load_async.return_value = None
        
        # Test different error types
        for error_class in [RateLimitError, APIError, APITimeoutError, APIConnectionError]:
            embedding_service.client.embeddings.create.side_effect = error_class("Test error")
            
            with pytest.raises(EmbeddingServiceError) as exc_info:
                embedding_service.create_embeddings(chunks)
            
            assert str(error_class.__name__) in str(exc_info.value)
            embedding_service.metrics.increment_counter.assert_called_with(
                f"openai_error_{error_class.__name__.lower()}"
            )

def test_cosine_similarity():
    """Test cosine similarity calculation."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    c = [1.0, 0.0]
    
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(a, c) == 1.0
    assert cosine_similarity(a, [-1.0, 0.0]) == -1.0

def test_create_book_data(embedding_service):
    """Test creation of BookDataInterface instance."""
    chunks = ["test chunk"]
    mock_embedding = [0.1] * 1536
    
    with patch('src.embedding.extract_dates') as mock_dates, \
         patch('src.embedding.extract_named_entities') as mock_entities, \
         patch('src.embedding.extract_key_phrases') as mock_phrases:
        
        mock_dates.return_value = []
        mock_entities.return_value = []
        mock_phrases.return_value = []
        
        embedding_service.create_embeddings.return_value = [mock_embedding]
        
        book_data = create_book_data(chunks, embedding_service)
        
        assert isinstance(book_data, BookDataInterface)
        assert book_data.get_chunks() == chunks
        assert book_data.get_embeddings() == [mock_embedding]

@pytest.mark.integration
def test_embedding_service_integration():
    """Integration test for EmbeddingService with real API."""
    service = EmbeddingService()
    test_chunks = ["Test text for embedding"]
    
    try:
        embeddings = service.create_embeddings(test_chunks)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
    except Exception as e:
        pytest.fail(f"Embedding creation failed: {str(e)}")
