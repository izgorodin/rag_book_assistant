import pytest
from src.text_processing import split_into_chunks

@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 50),
    (500, 100)
])
def test_split_into_chunks(chunk_size, overlap):
    # Generate a longer text
    text = "This is a test text. " * 100 

    chunks = split_into_chunks(text, chunk_size, overlap)

    assert isinstance(chunks, list), "Function should return a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    
    # Check that all chunks do not exceed the maximum size
    assert all(len(chunk) <= chunk_size for chunk in chunks), "Chunks should not exceed max size"
    
    # Check that most chunks (except the last one) are close to the maximum size
    assert all(len(chunk) >= chunk_size * 0.9 for chunk in chunks[:-1]), "Most chunks should be close to max size"
    
    # Check the overlap between chunks
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            assert chunks[i][-overlap:] == chunks[i+1][:overlap], f"Chunks {i} and {i+1} should overlap correctly"

    # Check that the entire original text is contained in the chunks
    reconstructed_text = "".join(chunks)
    assert text in reconstructed_text, "All original text should be contained in chunks"

    # Check that there is no significant text duplication
    assert len(reconstructed_text) <= len(text) * 1.5, "There should not be significant text duplication"