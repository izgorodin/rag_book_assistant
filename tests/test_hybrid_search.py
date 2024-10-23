import pytest
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from src.hybrid_search import HybridSearch

def load_test_data() -> Tuple[List[str], List[List[float]]]:
    base_path = 'data/embeddings'
    filename = 'book_chunks_embeddings.pkl'
    with open(os.path.join(base_path, filename), 'rb') as f:
        data = pickle.load(f)
    print("Type of data:", type(data))
    print("Keys in data:", data.keys() if isinstance(data, dict) else "No keys (not a dict)")
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    
    print("Type of chunks:", type(chunks))
    print("Type of embeddings:", type(embeddings))
    print("Number of chunks:", len(chunks))
    print("Number of embeddings:", len(embeddings))
    
    return chunks, embeddings

@pytest.fixture
def hybrid_search():
    chunks, embeddings = load_test_data()
    return HybridSearch(chunks, embeddings)

@pytest.fixture
def sample_chunks():
    return [
        "This is the first chunk about AI.",
        "The second chunk talks about machine learning.",
        "Third chunk discusses neural networks.",
        "Fourth chunk is about deep learning applications."
    ]

@pytest.fixture
def sample_embeddings():
    return [np.random.rand(1536) for _ in range(4)]  # 4 embeddings of size 1536

def test_search(hybrid_search):
    query = "Artificial Intelligence in modern applications"
    results = hybrid_search.search(query, top_k=5)
    assert len(results) == 5
    assert all('chunk' in result and 'score' in result for result in results)

def test_embedding_weight(hybrid_search):
    query = "Machine learning algorithms"
    results_default = hybrid_search.search(query, top_k=3)
    
    hybrid_search_high_bm25 = HybridSearch(hybrid_search.chunks, hybrid_search.embeddings, embedding_weight=0.1)
    results_high_bm25 = hybrid_search_high_bm25.search(query, top_k=3)
    
    assert results_default != results_high_bm25, "Results should differ with different embedding weights"

def test_hybrid_search_initialization(sample_chunks, sample_embeddings):
    hybrid_search = HybridSearch(sample_chunks, sample_embeddings)
    assert len(hybrid_search.chunks) == len(sample_chunks)
    assert len(hybrid_search.embeddings) == len(sample_embeddings)

def test_hybrid_search_search(sample_chunks, sample_embeddings):
    hybrid_search = HybridSearch(sample_chunks, sample_embeddings)
    results = hybrid_search.search("AI and machine learning", top_k=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all('chunk' in result and 'score' in result for result in results)

def test_hybrid_search_embedding_weight(sample_chunks, sample_embeddings):
    hybrid_search_default = HybridSearch(sample_chunks, sample_embeddings)
    hybrid_search_high_bm25 = HybridSearch(sample_chunks, sample_embeddings, embedding_weight=0.1)
    
    results_default = hybrid_search_default.search("AI", top_k=2)
    results_high_bm25 = hybrid_search_high_bm25.search("AI", top_k=2)
    
    assert results_default != results_high_bm25
    assert all(isinstance(result, dict) for result in results_default + results_high_bm25)
    assert all('chunk' in result and 'score' in result for result in results_default + results_high_bm25)

EMBEDDINGS_PATH = 'data/embeddings/test-file_chunks_embeddings.pkl'
