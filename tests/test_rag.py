import pytest
from src.rag import generate_answer, rag_query
from src.hybrid_search import HybridSearch
from src.book_data_interface import BookDataInterface

@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_find_most_relevant_chunks(sample_chunks, sample_embeddings, top_k):
    query = "Test query"
    hybrid_search = HybridSearch(sample_chunks, sample_embeddings)
    relevant_chunks = hybrid_search.search(query, top_k=top_k)
    
    assert isinstance(relevant_chunks, list), "Function should return a list"
    assert len(relevant_chunks) == top_k, f"Function should return {top_k} most relevant chunks"
    assert all(isinstance(chunk, dict) for chunk in relevant_chunks), "Each result should be a dictionary"
    assert all('chunk' in chunk and 'score' in chunk for chunk in relevant_chunks), "Each result should have 'chunk' and 'score' keys"

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
