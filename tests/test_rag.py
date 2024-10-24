import pytest
import time
from src.openai_service import OpenAIService
from src.rag import generate_answer, rag_query
from src.hybrid_search import HybridSearch
from src.book_data_interface import BookDataInterface
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_openai_service():
    with patch('src.rag.OpenAIService') as mock_service:
        mock_service.return_value.generate_answer.return_value = "Mocked answer"
        yield mock_service

@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_find_most_relevant_chunks(sample_chunks, sample_embeddings, top_k):
    query = "Test query"
    hybrid_search = HybridSearch(sample_chunks, sample_embeddings)
    relevant_chunks = hybrid_search.search(query, top_k=top_k)
    
    assert isinstance(relevant_chunks, list), "Function should return a list"
    assert len(relevant_chunks) == top_k, f"Function should return {top_k} most relevant chunks"
    assert all(isinstance(chunk, dict) for chunk in relevant_chunks), "Each result should be a dictionary"
    assert all('chunk' in chunk and 'score' in chunk for chunk in relevant_chunks)

@pytest.mark.parametrize("use_real_api", [True, False])
def test_generate_answer(openai_client, use_real_api):
    query = "Test query"
    context = "Test context"
    
    answer = generate_answer(query, context)
    
    assert isinstance(answer, str), "Function should return a string"
    assert len(answer) > 0, "Answer should not be empty"

@pytest.mark.parametrize("use_real_api", [True, False])
def test_rag_query(openai_client, sample_chunks, sample_embeddings, use_real_api):
    query = "Test query"
    book_data = BookDataInterface(sample_chunks, sample_embeddings, {})
    
    answer = rag_query(query, book_data)
    
    assert isinstance(answer, str), "Function should return a string"
    assert len(answer) > 0, "Answer should not be empty"

@pytest.mark.parametrize("query", [
    "Short query",
    "A very long query that contains multiple sentences and specific terms",
    ""
])
def test_rag_query_with_different_queries(query):
    # Implement test logic here
    assert isinstance(query, str), "Query should be a string"
    # Additional logic to test the behavior of rag_query with different queries can be added here

@pytest.mark.performance
def test_rag_query_performance():
    query = "Test query"
    book_data = BookDataInterface(["Test chunk"] * 1000, [[0.1] * 1536] * 1000, {})
    
    start_time = time.time()
    rag_query(query, book_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 10, f"RAG query took {execution_time} seconds, which is more than the expected 5 seconds"

def test_rag_error_handling():
    query = "Test query"
    book_data = BookDataInterface([], [], {})
    
    with pytest.raises(Exception):
        rag_query(query, book_data)

def test_generate_answer(mock_openai_service):
    query = "Test query"
    context = "Test context"
    
    answer = generate_answer(query, context)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    mock_openai_service.return_value.generate_answer.assert_called_once_with(query, context)

@pytest.mark.parametrize("error_type, error_message, expected_message", [
    (RateLimitError, "Rate limit exceeded", "Rate limit exceeded"),
    (APIError, "API error", "API error"),
    (APITimeoutError, "Request timed out", "Request timed out")
])
def test_generate_answer_error_handling(error_type, error_message, expected_message):
    with patch('src.rag.OpenAIService.generate_answer') as mock_generate:
        error = OpenAIService.create_test_exception(error_type, error_message)
        mock_generate.side_effect = error
        answer = generate_answer("Query", "Context")
        assert "Sorry, I encountered an error" in answer
        assert expected_message in answer
