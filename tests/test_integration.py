import pytest
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.mark.parametrize("use_real_api", [True, False])
def test_different_book_types(tmp_path, patch_openai, use_real_api):
    logger.debug(f"Starting test_different_book_types with use_real_api={use_real_api}")
    # Create temporary test books
    fiction_book = tmp_path / "fiction.txt"
    fiction_book.write_text("This is a fiction book about wizards and magic.")
    
    non_fiction_book = tmp_path / "non_fiction.txt"
    non_fiction_book.write_text("This is a non-fiction book about the history of science.")
    
    for book in [fiction_book, non_fiction_book]:
        logger.debug(f"Processing book: {book}")
        text = load_and_preprocess_text(str(book))
        logger.debug(f"Preprocessed text: {text[:50]}...")
        
        chunks = split_into_chunks(text)
        logger.debug(f"Split into {len(chunks)} chunks")
        
        logger.debug("Creating embeddings")
        embeddings = create_embeddings(chunks)
        logger.debug(f"Created {len(embeddings)} embeddings")
        
        query = "What is this book about?"
        logger.debug(f"Querying: {query}")
        answer = rag_query(query, chunks, embeddings)
        logger.debug(f"Received answer: {answer}")
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        if "fiction" in book.name:
            assert any(word.lower() in answer.lower() for word in ["fiction", "wizard", "magic"]), f"Answer should be relevant to the fiction book. Got: {answer}"
        else:
            assert any(word.lower() in answer.lower() for word in ["non-fiction", "history", "science"]), f"Answer should be relevant to the non-fiction book. Got: {answer}"
    
    logger.debug("Test completed successfully")

@pytest.mark.parametrize("use_real_api", [True, False])
def test_rag_query(openai_client, sample_text, use_real_api):
    text = load_and_preprocess_text("tests/test_book.txt")
    chunks = split_into_chunks(text)
    embeddings = create_embeddings(chunks)
    
    query = "What are the main laws mentioned in the book?"
    answer = rag_query(query, chunks, embeddings)
    
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"
    assert "Law" in answer, "Answer should reference laws from the book"