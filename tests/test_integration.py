import pytest
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query
from tests.conftest import run_with_and_without_api

@run_with_and_without_api
def test_different_book_types(tmp_path, patch_openai, use_api):
    # Create temporary test books
    fiction_book = tmp_path / "fiction.txt"
    fiction_book.write_text("This is a fiction book about wizards and magic.")
    
    non_fiction_book = tmp_path / "non_fiction.txt"
    non_fiction_book.write_text("This is a non-fiction book about the history of science.")
    
    for book in [fiction_book, non_fiction_book]:
        text = load_and_preprocess_text(str(book))
        chunks = split_into_chunks(text)
        embeddings = create_embeddings(chunks)
        
        query = "What is this book about?"
        answer = rag_query(query, chunks, embeddings)
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        if "fiction" in book.name:
            assert any(word in answer.lower() for word in ["fiction", "wizard", "magic"]), "Answer should be relevant to the fiction book"
        else:
            assert any(word in answer.lower() for word in ["non-fiction", "history", "science"]), "Answer should be relevant to the non-fiction book"