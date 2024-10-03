import pytest
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query

@pytest.fixture(scope="module")
def prepared_data():
    text = load_and_preprocess_text("tests/test_book.txt")
    chunks = split_into_chunks(text)
    embeddings = create_embeddings(chunks)
    return chunks, embeddings

def test_full_pipeline(prepared_data):
    chunks, embeddings = prepared_data
    
    queries = [
        "What is the main topic of the book?",
        "Who is the protagonist?",
        "What happens in the climax of the story?"
    ]
    
    for query in queries:
        answer = rag_query(query, chunks, embeddings)
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        # Add more specific assertions based on the content of your test book

def test_different_book_types(tmp_path):
    # Create temporary test books
    book1 = tmp_path / "fiction.txt"
    book1.write_text("This is a fiction book about wizards and magic.")
    
    book2 = tmp_path / "non_fiction.txt"
    book2.write_text("This is a non-fiction book about the history of science.")
    
    for book in [book1, book2]:
        text = load_and_preprocess_text(str(book))
        chunks = split_into_chunks(text)
        embeddings = create_embeddings(chunks)
        
        query = "What is this book about?"
        answer = rag_query(query, chunks, embeddings)
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        if "fiction" in book.name:
            assert "wizard" in answer.lower() or "magic" in answer.lower(), "Answer should be relevant to the fiction book"
        else:
            assert "science" in answer.lower() or "history" in answer.lower(), "Answer should be relevant to the non-fiction book"