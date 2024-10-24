import pytest
from src.pinecone_manager import PineconeManager
from tests.mock_pinecone import MockPinecone
import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def pinecone_manager():
    mock_pinecone = MockPinecone(api_key="fake_key", error_probability=0.5, max_consecutive_errors=2)
    manager = PineconeManager(index_name="test-index", pinecone_client=mock_pinecone,
                              max_retries=5, min_wait=1, max_wait=3)
    yield manager
    try:
        mock_pinecone.reset()
        manager.clear_index()
    except Exception as e:
        logger.warning(f"Error during teardown: {str(e)}")

def test_pinecone_manager_initialization(pinecone_manager):
    assert pinecone_manager.is_available(), "Pinecone manager should be available"

@pytest.mark.parametrize("chunks,embeddings", [
    (["Test chunk 1", "Test chunk 2"], [[0.1] * 1536, [0.2] * 1536]),
    ([], []),
    (["Very long chunk" * 1000], [[0.3] * 1536])
])
def test_upsert_and_search(pinecone_manager, chunks, embeddings):
    pinecone_manager.upsert_embeddings(chunks, embeddings)
    
    if chunks:
        results = pinecone_manager.search_similar(embeddings[0], top_k=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0]["chunk"].startswith(chunks[0][:1000]), f"Expected chunk to start with '{chunks[0][:1000]}', got '{results[0]['chunk']}'"
    else:
        results = pinecone_manager.search_similar([0.1] * 1536, top_k=1)
        assert len(results) == 0, f"Expected 0 results for empty index, got {len(results)}"

def test_get_or_create_embeddings(pinecone_manager):
    chunks = ["Test chunk 3", "Test chunk 4"]
    
    def mock_embedding_function(x):
        return [[0.1] * 1536 for _ in x]
    
    embeddings = pinecone_manager.get_or_create_embeddings(chunks, mock_embedding_function)
    
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert all(len(emb) == 1536 for emb in embeddings), "All embeddings should have length 1536"

def test_clear_index(pinecone_manager):
    chunks = ["Test chunk 5"]
    embeddings = [[0.1] * 1536]
    
    pinecone_manager.upsert_embeddings(chunks, embeddings)
    pinecone_manager.clear_index()
    
    results = pinecone_manager.search_similar(embeddings[0], top_k=1)
    assert len(results) == 0, f"Expected 0 results after clearing index, got {len(results)}"

def test_error_handling(pinecone_manager):
    with pytest.raises(ValueError, match="Pinecone index is not available"):
        pinecone_manager.index = None
        pinecone_manager.upsert_embeddings(["Test"], [[0.1] * 1536])

@pytest.mark.parametrize("chunk_count", [1, 10, 100])
def test_batch_operation(pinecone_manager, chunk_count):
    chunks = [f"Chunk {i}" for i in range(chunk_count)]
    embeddings = [[0.1] * 1536 for _ in range(chunk_count)]
    
    with pinecone_manager.batch_operation():
        pinecone_manager.upsert_embeddings(chunks, embeddings)
    
    results = pinecone_manager.search_similar(embeddings[0], top_k=chunk_count)
    assert len(results) == chunk_count, f"Expected {chunk_count} results, got {len(results)}"
