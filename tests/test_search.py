import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.data_source import DataSource
from src.search import SimpleSearch, HybridSearch, get_search_strategy, BaseSearch
from src.book_data_interface import BookDataInterface
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

@pytest.fixture
def mock_data_source():
    mock = Mock(spec=DataSource)
    mock.get_chunks.return_value = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "All that glitters is not gold"
    ]
    mock.get_embeddings.return_value = [
        np.array([0.1, 0.2, 0.3] * (1536 // 3)),
        np.array([0.4, 0.5, 0.6] * (1536 // 3)),
        np.array([0.7, 0.8, 0.9] * (1536 // 3)),
        np.array([0.2, 0.3, 0.4] * (1536 // 3))
    ]
    return mock

def test_simple_search(mock_data_source):
    search = SimpleSearch(mock_data_source)
    results = search.search("journey", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert 'chunk' in results[0] and 'score' in results[0]

def test_hybrid_search(mock_data_source):
    search = HybridSearch(mock_data_source)
    results = search.search("journey", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert 'chunk' in results[0] and 'score' in results[0]

def test_get_search_strategy(mock_data_source):
    simple_strategy = get_search_strategy("simple", mock_data_source)
    assert isinstance(simple_strategy, SimpleSearch)

    hybrid_strategy = get_search_strategy("hybrid", mock_data_source)
    assert isinstance(hybrid_strategy, HybridSearch)

    with pytest.raises(ValueError):
        get_search_strategy("unknown", mock_data_source)

@patch('src.search.create_embeddings')
def test_search_error_handling(mock_create_embeddings, mock_data_source):
    mock_create_embeddings.side_effect = Exception("Test error")
    search = HybridSearch(mock_data_source)
    
    with pytest.raises(RAGError) as exc_info:
        search.search("query")
    
    assert "Test error" in str(exc_info.value)

@patch('src.search.create_embeddings')
def test_embedding_scores(mock_create_embeddings, mock_data_source):
    mock_create_embeddings.return_value = [np.array([0.2, 0.3, 0.4] * (1536 // 3))]
    search = HybridSearch(mock_data_source)
    scores = search._get_embedding_scores("journey")
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(mock_data_source.get_chunks())

def test_get_top_chunks(mock_data_source):
    search = SimpleSearch(mock_data_source)
    scores = np.array([0.9, 0.7, 0.8, 0.6])
    results = search._get_top_chunks(scores, top_k=2)
    assert len(results) == 2
    assert results[0]['score'] > results[1]['score']

@patch('src.search.create_embeddings')
def test_simple_search_error_handling(mock_create_embeddings, mock_data_source):
    mock_create_embeddings.side_effect = Exception("Test error")
    search = SimpleSearch(mock_data_source)
    
    with pytest.raises(RAGError) as exc_info:
        search.search("query")
    
    assert "Test error" in str(exc_info.value)

@patch('src.search.create_embeddings')
def test_hybrid_search_error_handling(mock_create_embeddings, mock_data_source):
    mock_create_embeddings.side_effect = Exception("Test error")
    search = HybridSearch(mock_data_source)
    
    with pytest.raises(RAGError) as exc_info:
        search.search("query")
    
    assert "Test error" in str(exc_info.value)
