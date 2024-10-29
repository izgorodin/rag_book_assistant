import pytest
from src.embedding import EmbeddingService
from tests.conftest import (
    TEST_EMBEDDING_DIM,
    TEST_EMBEDDING_VALUES
)

# def test_create_embeddings_with_cache(mock_openai_client, mock_cache, sample_text, caplog):
#     """Test embedding creation with caching and logging."""
#     service = EmbeddingService(mock_openai_client, mock_cache)
#     texts = [sample_text]
#     
#     embeddings = service.create_embeddings(texts)
#     
#     assert len(embeddings) == 1
#     assert len(embeddings[0]) == TEST_EMBEDDING_DIM
#     assert embeddings[0] == TEST_EMBEDDING_VALUES * (TEST_EMBEDDING_DIM // len(TEST_EMBEDDING_VALUES))
#     
#     assert "Creating embeddings for 1 texts" in caplog.text
#     assert "Processed batch 1: 1/1" in caplog.text
