import pytest
from src.rag import find_most_relevant_chunks, generate_answer, rag_query
from tests.test_utils import patch_openai, run_with_and_without_api, sample_chunks, sample_embeddings

@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_find_most_relevant_chunks(sample_chunks, sample_embeddings, top_k):
    query = "Test query"
    query_embedding = [0.15] * 1536
    
    relevant_chunks = find_most_relevant_chunks(query, query_embedding, sample_chunks, sample_embeddings, top_k)
    
    assert isinstance(relevant_chunks, list), "Function should return a list"
    assert len(relevant_chunks) == top_k, f"Function should return {top_k} most relevant chunks"
    assert all(chunk in sample_chunks for chunk in relevant_chunks), "Returned chunks should be from the original chunks"

@pytest.mark.usefixtures("patch_openai")
@run_with_and_without_api
def test_generate_answer(use_api):
    query = "Test query"
    context = "Test context"
    
    answer = generate_answer(query, context)
    
    assert isinstance(answer, str), "Function should return a string"
    assert len(answer) > 0, "Answer should not be empty"

@pytest.mark.usefixtures("patch_openai")
@run_with_and_without_api
def test_rag_query(sample_chunks, sample_embeddings, use_api):
    query = "Test query"
    
    answer = rag_query(query, sample_chunks, sample_embeddings)
    
    assert isinstance(answer, str), "Function should return a string"
    assert len(answer) > 0, "Answer should not be empty"