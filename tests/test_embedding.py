import pytest
from src.embedding import create_embeddings
from tests.test_utils import patch_openai, run_with_and_without_api, sample_chunks

@pytest.mark.usefixtures("patch_openai")
@run_with_and_without_api
def test_create_embeddings(sample_chunks, use_api):
    embeddings = create_embeddings(sample_chunks)
    
    assert isinstance(embeddings, list), "Function should return a list"
    assert len(embeddings) == len(sample_chunks), "Number of embeddings should match number of chunks"
    assert all(len(emb) == 1536 for emb in embeddings), "Each embedding should have 1536 dimensions"

@pytest.mark.usefixtures("patch_openai")
@run_with_and_without_api
def test_create_embeddings_api_error(sample_chunks, use_api, patch_openai):
    if not use_api:
        patch_openai.embeddings.create.side_effect = Exception("API Error")
    
    with pytest.raises(Exception):
        create_embeddings(sample_chunks)