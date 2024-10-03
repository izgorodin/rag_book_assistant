import pytest
from unittest.mock import patch
from src.rag import find_most_relevant_chunks, generate_answer, rag_query

@pytest.fixture
def sample_chunks():
    return [
        "This is the first chunk about apples.",
        "This is the second chunk about bananas.",
        "This is the third chunk about oranges."
    ]

@pytest.fixture
def sample_embeddings():
    return [
        [0.1] * 1536,
        [0.2] * 1536,
        [0.3] * 1536
    ]

def test_find_most_relevant_chunks(sample_chunks, sample_embeddings):
    query = "Tell me about apples"
    query_embedding = [0.15] * 1536
    
    relevant_chunks = find_most_relevant_chunks(query, query_embedding, sample_chunks, sample_embeddings, 2)
    
    assert isinstance(relevant_chunks, list), "Function should return a list"
    assert len(relevant_chunks) == 2, "Function should return 2 most relevant chunks"
    assert sample_chunks[0] in relevant_chunks, "Most relevant chunk should be included"

@patch("openai.ChatCompletion.create")
def test_generate_answer(mock_create):
    mock_create.return_value = {
        "choices": [{"message": {"content": "This is a test answer about apples."}}]
    }
    
    query = "Tell me about apples"
    context = "Apples are red fruits."
    
    answer = generate_answer(query, context)
    
    assert isinstance(answer, str), "Function should return a string"
    assert "apples" in answer.lower(), "Answer should be relevant to the query"

@patch("src.rag.find_most_relevant_chunks")
@patch("src.rag.generate_answer")
def test_rag_query(mock_generate_answer, mock_find_chunks, sample_chunks, sample_embeddings):
    mock_find_chunks.return_value = sample_chunks[:2]
    mock_generate_answer.return_value = "This is a test answer about fruits."
    
    query = "Tell me about fruits"
    answer = rag_query(query, sample_chunks, sample_embeddings)
    
    assert isinstance(answer, str), "Function should return a string"
    assert "fruits" in answer.lower(), "Answer should be relevant to the query"
