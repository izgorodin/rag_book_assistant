import pytest
import os
from src.text_processing import load_and_preprocess_text, split_into_chunks

def test_load_and_preprocess_text():
    test_file_path = os.path.join(os.path.dirname(__file__), 'data', 'test_book.txt')
    with open(test_file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Test with raw text
    processed_text = load_and_preprocess_text(text_content)
    _verify_processed_text(processed_text)
    
    # Test with already processed text
    reprocessed_text = load_and_preprocess_text(processed_text)
    _verify_processed_text(reprocessed_text)

def _verify_processed_text(processed_text):
    assert isinstance(processed_text, dict), "Processed text should be a dictionary"
    assert 'text' in processed_text, "Processed text should contain 'text' key"
    assert isinstance(processed_text['text'], str), "The 'text' value should be a string"
    assert len(processed_text['text']) > 0, "Processed text should not be empty"
    assert 'chunks' in processed_text, "Processed text should contain 'chunks' key"
    assert isinstance(processed_text['chunks'], list), "Chunks should be a list"

@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 50),
    (500, 100)
])
def test_split_into_chunks(chunk_size, overlap):
    text = ' '.join(['word'] * 1000)  # Create a sample text
    chunks = split_into_chunks(text, chunk_size, overlap)

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

def test_split_into_chunks_with_dict_input():
    text_dict = {'text': ' '.join(['word'] * 1000)}
    chunks = split_into_chunks(text_dict, chunk_size=100, overlap=20)
    assert isinstance(chunks, list), "Function should return a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"

def _verify_book_data(book_data):
    # Test basic properties
    assert len(book_data.get_chunks()) > 0
    assert len(book_data.get_embeddings()) == len(book_data.get_chunks())
    assert all(len(emb) == 1536 for emb in book_data.get_embeddings())
    # Здесь можно добавить более конкретные проверки, если содержание test_book.txt известно
    # Test new features
    assert isinstance(book_data.get_dates(), list)
    assert isinstance(book_data.get_entities(), list)
    assert isinstance(book_data.get_key_phrases(), list)
