import pytest
import os
import pickle
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
