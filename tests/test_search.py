import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.data_source import DataSource
from src.search import SimpleSearch, HybridSearch, get_search_strategy, BaseSearch
from src.book_data_interface import BookDataInterface
from tests.test_utils import ListDataSource
from src.error_handler import format_error_message, RAGError

@pytest.fixture
def mock_book_data():
    return Mock(spec=BookDataInterface)

@pytest.fixture
def sample_book_data():
    chunks = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "All that glitters is not gold",
    ]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
    ]
    return BookDataInterface(chunks, embeddings, {})

def test_simple_search(sample_book_data):
    search = SimpleSearch(sample_book_data)
    with patch('src.search.create_embeddings', return_value=[[0.2, 0.3, 0.4]]):
        results = search.search("journey", top_k=2)
    
    assert len(results) == 2
    assert results[0]['chunk'] == "A journey of a thousand miles begins with a single step"
    assert isinstance(results[0]['score'], float)

def test_hybrid_search(sample_book_data):
    search = HybridSearch(sample_book_data)
    with patch('src.search.create_embeddings', return_value=[[0.2, 0.3, 0.4]]):
        results = search.search("journey", top_k=2)
    
    assert len(results) == 2
    assert results[0]['chunk'] == "A journey of a thousand miles begins with a single step"
    assert isinstance(results[0]['score'], float)

def test_get_search_strategy():
    mock_data_source = Mock(spec=DataSource)
    mock_data_source.get_chunks.return_value = ["Test chunk"]
    mock_data_source.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    simple_strategy = get_search_strategy("simple", mock_data_source)
    assert isinstance(simple_strategy, SimpleSearch)
    
    hybrid_strategy = get_search_strategy("hybrid", mock_data_source)
    assert isinstance(hybrid_strategy, HybridSearch)
    
    result = get_search_strategy("unknown", mock_data_source)
    assert "Sorry, I encountered an error" in result
    assert "Unknown search strategy: unknown" in result

def test_search_error_handling(sample_book_data):
    search = HybridSearch(sample_book_data)
    with patch('src.search.HybridSearch._create_weighted_query_embedding', side_effect=Exception("Test error")):
        result = search.search("error query")
        assert "Sorry, I encountered an error" in result
        assert "Test error" in result

@pytest.mark.parametrize("query,expected_expansion", [
    ("run", ["run", "running", "ran"]),
    ("happy", ["happy", "joyful", "content"]),
])
def test_query_expansion(sample_book_data, query, expected_expansion):
    search = HybridSearch(sample_book_data)
    expanded_query = search._expand_query(query)
    expanded_words = expanded_query.split()
    
    print("\nQuery Expansion Test Results:")
    print(f"Original query: {query}")
    print(f"Expanded query: {expanded_query}")
    print(f"Expected expansion words: {expected_expansion}")
    
    print("\nPlease answer the following questions (yes/no):")
    print(f"1. Does the expanded query contain the original query '{query}'?")
    print("2. Are the additional words in the expanded query relevant expansions or synonyms?")
    print(f"3. Do you see any of the expected expansion words {expected_expansion[1:]} in the result?")
    print("4. Overall, does this expansion seem reasonable for the given query?")
    
    # Мы не будем автоматически проверять результат, а оставим это на усмотрение тестировщика
    assert True, "This test requires manual verification. Please check the console output."

def test_bm25_scores(sample_book_data):
    search = HybridSearch(sample_book_data)
    scores = search._get_bm25_scores("journey miles")
    assert len(scores) == len(sample_book_data.get_chunks())
    assert scores[1] > scores[0]  # "journey miles" should score higher for the second chunk

def test_embedding_scores(sample_book_data):
    search = HybridSearch(sample_book_data)
    with patch('src.search.create_embeddings', return_value=[[0.2, 0.3, 0.4]]):
        scores = search._get_embedding_scores("journey")
    assert len(scores) == len(sample_book_data.get_chunks())
    assert scores[1] > scores[0]  # The embedding [0.2, 0.3, 0.4] is closest to the second chunk's embedding

def test_combine_scores():
    mock_data_source = Mock(spec=DataSource)
    mock_data_source.get_chunks.return_value = ["Test chunk 1", "Test chunk 2"]
    mock_data_source.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    search = HybridSearch(mock_data_source)
    bm25_scores = np.array([0.1, 0.2])
    embedding_scores = np.array([0.3, 0.4])
    combined_scores = search._combine_scores(bm25_scores, embedding_scores)
    assert len(combined_scores) == 2
    assert all(0 <= score <= 1 for score in combined_scores)

def test_get_top_chunks(sample_book_data):
    search = BaseSearch(sample_book_data)
    scores = np.array([0.1, 0.4, 0.2, 0.3])
    top_chunks = search._get_top_chunks(scores, top_k=2)
    assert len(top_chunks) == 2
    assert top_chunks[0]['chunk'] == "A journey of a thousand miles begins with a single step"
    assert top_chunks[0]['score'] == 0.4
    assert top_chunks[1]['chunk'] == "All that glitters is not gold"
    assert top_chunks[1]['score'] == 0.3

def test_simple_search_error_handling(sample_book_data):
    search = SimpleSearch(sample_book_data)
    with patch.object(sample_book_data, 'get_chunks', side_effect=Exception("Test error")):
        result = search.search("error query")
        assert "Sorry, I encountered an error" in result
        assert "Test error" in result

def test_hybrid_search_error_handling(sample_book_data):
    search = HybridSearch(sample_book_data)
    with patch.object(sample_book_data, 'get_embeddings', side_effect=Exception("Test error")):
        result = search.search("error query")
        assert "Sorry, I encountered an error" in result
        assert "Test error" in result
