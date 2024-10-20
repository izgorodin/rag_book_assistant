import pytest
from unittest.mock import patch, Mock
from src.embedding import create_embeddings

@pytest.mark.parametrize("chunks, expected_length", [
    (["Test chunk 1", "Test chunk 2"], 2),
    (["Single chunk"], 1),
    ([], 0)
])
def test_create_embeddings_length(chunks, expected_length):
    mock_embedding = [0.1] * 1536
    with patch('src.embedding.client.embeddings.create', return_value=Mock(data=[Mock(embedding=mock_embedding)])):
        embeddings = create_embeddings(chunks)
        assert len(embeddings) == expected_length, f"Expected {expected_length} embeddings, got {len(embeddings)}"

def test_create_embeddings_structure():
    chunks = ["Test chunk"]
    mock_embedding = [0.1] * 1536
    with patch('src.embedding.client.embeddings.create', return_value=Mock(data=[Mock(embedding=mock_embedding)])):
        embeddings = create_embeddings(chunks)
        assert isinstance(embeddings, list), "Embeddings should be a list"
        assert isinstance(embeddings[0], list), "Each embedding should be a list"
        assert len(embeddings[0]) == 1536, "Each embedding should have 1536 dimensions"

def test_create_embeddings_api_error():
    chunks = ["Test chunk"]
    with patch('src.embedding.client.embeddings.create', side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            create_embeddings(chunks)
        assert "Error creating embedding" in str(exc_info.value)

@pytest.mark.parametrize("chunk,expected_length", [
    ("", 0),
    ("Short text", 1536),
    ("A" * 1000, 1536)
])
def test_create_embeddings_various_lengths(chunk, expected_length):
    mock_embedding = [0.1] * expected_length
    with patch('src.embedding.client.embeddings.create', return_value=Mock(data=[Mock(embedding=mock_embedding)])):
        embeddings = create_embeddings([chunk])
        assert len(embeddings) == 1, "Should create one embedding regardless of chunk length"
        assert len(embeddings[0]) == expected_length, f"Embedding should always have {expected_length} dimensions"