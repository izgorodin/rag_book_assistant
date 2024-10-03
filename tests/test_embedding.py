import pytest
from unittest.mock import patch
from src.embedding import create_embeddings

@pytest.fixture
def mock_openai_response():
    return {
        "data": [
            {
                "embedding": [0.1] * 1536  # Mocked embedding vector
            }
        ]
    }

@patch("openai.Embedding.create")
def test_create_embeddings(mock_create, mock_openai_response):
    mock_create.return_value = mock_openai_response
    
    chunks = ["This is chunk 1", "This is chunk 2"]
    embeddings = create_embeddings(chunks)
    
    assert isinstance(embeddings, list), "Function should return a list"
    assert len(embeddings) == len(chunks), "Number of embeddings should match number of chunks"
    assert all(len(emb) == 1536 for emb in embeddings), "Each embedding should have 1536 dimensions"

@patch("openai.OpenAI")
def test_create_embeddings_api_error(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.embeddings.create.side_effect = Exception("API Error")
    
    chunks = ["This is a test chunk"]
    with pytest.raises(Exception):
        create_embeddings(chunks)
