import pytest
from src.text_processing import load_and_preprocess_text, split_into_chunks
from tests.test_utils import sample_text

def test_load_and_preprocess_text(tmp_path):
    """
    Test the load_and_preprocess_text function.
    """
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test   text with some special characters: @#$%^&*()!")
    
    processed_text = load_and_preprocess_text(str(test_file))
    
    assert isinstance(processed_text, str), "Function should return a string"
    assert "This is a test text with some special characters" in processed_text, "Processed text should contain the original content without special characters"
    assert "@#$%^&*()" not in processed_text, "Processed text should not contain special characters"

@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 50),
    (500, 100)
])
def test_split_into_chunks(sample_text, chunk_size, overlap):
    """
    Test the split_into_chunks function with different chunk sizes and overlaps.
    """
    chunks = split_into_chunks(sample_text, chunk_size, overlap)

    assert isinstance(chunks, list), "Function should return a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    
    # Check that all chunks do not exceed the maximum size
    assert all(len(chunk.split()) <= chunk_size for chunk in chunks), "Chunks should not exceed max size"
    
    # Check that most chunks (except the last one) are close to the maximum size
    assert all(len(chunk.split()) >= chunk_size * 0.9 for chunk in chunks[:-1]), "Most chunks should be close to max size"
    
    # Check the overlap between chunks
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            assert chunks[i].split()[-overlap:] == chunks[i+1].split()[:overlap], f"Chunks {i} and {i+1} should overlap correctly"

def test_split_into_chunks_short_text():
    """
    Test the split_into_chunks function with a short text.
    """
    short_text = "Short text."
    chunks = split_into_chunks(short_text, chunk_size=10, overlap=2)
    
    assert len(chunks) == 1, "Should return one chunk for short text"
    assert chunks[0] == short_text, "Chunk should match the original text"