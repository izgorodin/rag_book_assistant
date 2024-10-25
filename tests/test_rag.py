import pytest
import time
from src.openai_service import OpenAIService
from src.rag import generate_answer, rag_query
from src.search import HybridSearch, SimpleSearch
from src.book_data_interface import BookDataInterface
from unittest.mock import patch, MagicMock, Mock
from src.error_handler import format_error_message, RAGError

@pytest.fixture
def mock_openai_service():
    return MagicMock(spec=OpenAIService)

@pytest.fixture
def mock_book_data():
    return Mock(spec=BookDataInterface)

@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_find_most_relevant_chunks(sample_chunks, sample_embeddings, top_k):
    query = "Test query"
    book_data = BookDataInterface(sample_chunks, sample_embeddings, {})
    hybrid_search = HybridSearch(book_data)
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
def test_rag_query_performance(mock_openai_service):
    query = "Test query"
    book_data = BookDataInterface(["Test chunk"] * 1000, [[0.1] * 1536] * 1000, {})
    
    start_time = time.time()
    rag_query(query, book_data, mock_openai_service)
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 10, f"RAG query took {execution_time} seconds, which is more than the expected 10 seconds"

def test_rag_error_handling(mock_openai_service):
    query = "Test query"
    book_data = BookDataInterface([], [], {})
    
    mock_openai_service.generate_answer.side_effect = RAGError("Test error")
    
    answer = rag_query(query, book_data, mock_openai_service)
    expected_error_message = format_error_message(RAGError("Test error"))
    assert expected_error_message in answer

def test_generate_answer(mock_openai_service):
    query = "Test query"
    context = "Test context"
    
    mock_openai_service.generate_answer.return_value = "Mocked answer"
    answer = generate_answer(query, context, mock_openai_service)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    mock_openai_service.generate_answer.assert_called_once_with(query, context)

@pytest.mark.parametrize("error_type, error_message", [
    (RAGError, "Test error"),
    (Exception, "Unexpected error"),
])
def test_generate_answer_error_handling(mock_openai_service, error_type, error_message):
    mock_openai_service.generate_answer.side_effect = error_type(error_message)
    answer = generate_answer("Query", "Context", mock_openai_service)
    expected_error_message = format_error_message(error_type(error_message))
    assert expected_error_message in answer

@patch('src.rag.PineconeManager')
@patch('src.rag.create_embeddings')
def test_rag_query(mock_create_embeddings, mock_pinecone_manager, mock_openai_service):
    query = "Test query"
    book_data = BookDataInterface([], [], {})
    
    mock_create_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_pinecone_manager.return_value.search_similar.return_value = [
        {"chunk": "Test chunk", "score": 0.9}
    ]
    mock_openai_service.generate_answer.return_value = "Test answer"
    
    answer = rag_query(query, book_data, mock_openai_service)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    mock_openai_service.generate_answer.assert_called_once()

def test_rag_query(mock_book_data, mock_openai_service):
    mock_book_data.chunks = ["Test chunk 1", "Test chunk 2"]
    mock_book_data.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    
    mock_openai_service.generate_answer.return_value = "Test answer"
    
    with patch('src.rag.get_search_strategy') as mock_get_strategy:
        mock_search = Mock(spec=SimpleSearch)
        mock_search.search.return_value = [{'chunk': 'Test chunk 1', 'score': 0.9}]
        mock_get_strategy.return_value = mock_search
        
        result = rag_query("Test question", mock_book_data, mock_openai_service)
        
        assert result == "Test answer"
        mock_search.search.assert_called_once_with("Test question")
        mock_openai_service.generate_answer.assert_called_once()

def test_rag_query_error_handling(mock_book_data, mock_openai_service):
    with patch('src.rag.get_search_strategy', side_effect=RAGError("Test error")):
        result = rag_query("Test question", mock_book_data, mock_openai_service)
        expected_error_message = format_error_message(RAGError("Test error"))
        assert expected_error_message in result
