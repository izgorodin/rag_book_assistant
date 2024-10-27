import pytest
from unittest.mock import Mock, patch, call
import numpy as np
from src.embedding import EmbeddingService, cosine_similarity, create_book_data
from src.book_data_interface import BookDataInterface

@pytest.fixture
def embedding_service():
    """Create EmbeddingService with mocked dependencies."""
    mock_openai = Mock()
    mock_vector_store = Mock()
    mock_cache_manager = Mock()
    return EmbeddingService(
        openai_client=mock_openai,
        vector_store=mock_vector_store,
        cache_manager=mock_cache_manager,
        batch_size=2
    )

class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    def test_create_embeddings_batch_processing(self, embedding_service):
        """Test that chunks are processed in correct batch sizes."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        mock_embedding = [0.1] * 1536
        
        # Mock cache miss for all chunks
        embedding_service.cache_manager.load.return_value = None
        
        # Mock OpenAI response
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_embedding)] * 2  # First batch
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Check batch processing
        assert embedding_service.client.embeddings.create.call_count == 2
        calls = embedding_service.client.embeddings.create.call_args_list
        assert calls[0][1]['input'] == ["chunk1", "chunk2"]  # First batch
        assert calls[1][1]['input'] == ["chunk3"]  # Second batch

    def test_create_embeddings_caching(self, embedding_service):
        """Test caching behavior for embeddings."""
        chunks = ["cached", "uncached"]
        mock_cached = [0.1] * 1536
        mock_new = [0.2] * 1536
        
        # Mock cache hit for first chunk, miss for second
        embedding_service.cache_manager.load.side_effect = [mock_cached, None]
        
        # Mock OpenAI response for uncached chunk
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_new)]
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Verify cache usage
        assert embeddings[0] == mock_cached  # Used cached embedding
        assert embeddings[1] == mock_new  # Created new embedding
        assert embedding_service.client.embeddings.create.call_count == 1

    def test_create_embeddings_pinecone_storage(self, embedding_service):
        """Test that vectors are properly stored in Pinecone."""
        chunks = ["test1", "test2"]
        mock_embedding = [0.1] * 1536
        
        embedding_service.cache_manager.load.return_value = None
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_embedding)] * 2
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Verify Pinecone storage
        expected_vectors = [
            {
                "id": "0",
                "values": mock_embedding,
                "metadata": {"text": "test1"}
            },
            {
                "id": "1",
                "values": mock_embedding,
                "metadata": {"text": "test2"}
            }
        ]
        embedding_service.vector_store.upsert_vectors.assert_called_once_with(expected_vectors)

    def test_create_embeddings_empty_chunks(self, embedding_service):
        """Test handling of empty chunks."""
        chunks = ["", "  ", "valid"]
        mock_embedding = [0.1] * 1536
        
        embedding_service.cache_manager.load.return_value = None
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_embedding)]
        )
        
        embeddings = embedding_service.create_embeddings(chunks)
        
        # Only valid chunk should be processed
        assert embedding_service.client.embeddings.create.call_count == 1
        assert len(embeddings) == 3

    def test_create_embeddings_dimension_check(self, embedding_service):
        """Test dimension validation for embeddings."""
        chunks = ["test"]
        wrong_dim_embedding = [0.1] * 100  # Wrong dimension
        
        embedding_service.cache_manager.load.return_value = None
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=wrong_dim_embedding)]
        )
        
        with pytest.raises(ValueError, match="Incorrect embedding dimension"):
            embedding_service.create_embeddings(chunks)

    def test_create_embedding_single(self, embedding_service):
        """Test creating embedding for single text."""
        text = "test"
        mock_embedding = [0.1] * 1536
        
        embedding_service.cache_manager.load.return_value = None
        embedding_service.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=mock_embedding)]
        )
        
        embedding = embedding_service.create_embedding(text)
        
        assert len(embedding) == 1536
        embedding_service.cache_manager.save.assert_called_once_with(text, mock_embedding)

def test_cosine_similarity():
    """Test cosine similarity calculation."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    c = [1.0, 0.0]
    
    assert cosine_similarity(a, b) == 0.0  # Perpendicular vectors
    assert cosine_similarity(a, c) == 1.0  # Same direction
    assert cosine_similarity(a, [-1.0, 0.0]) == -1.0  # Opposite direction

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