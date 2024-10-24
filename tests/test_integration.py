import pytest
import os
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query
from tests.ford_pinto_qa_data import qa_pairs
from src.pinecone_manager import PineconeManager
from tests.mock_pinecone import MockPinecone
from unittest.mock import MagicMock
from src.book_data_interface import BookDataInterface

from src.logger import setup_logger

logger = setup_logger()

# Путь к директории с тестовыми файлами
TEST_FILES_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.parametrize("book_type", ["fiction", "non-fiction", "scientific"])
def test_different_book_types(book_type, use_openai, tmp_path):
    logger.debug(f"Starting test_different_book_types with book_type={book_type}")
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

def test_different_book_types(use_openai):
    book_types = ["fiction", "non-fiction", "scientific"]
    for book_type in book_types:
        chunks, embeddings = process_book(f"sample_{book_type}_book.txt", use_openai=use_openai)
        assert len(chunks) > 0
        assert len(embeddings) == len(chunks)
        assert all(len(embedding) == 1536 for embedding in embeddings)

def test_rag_query(use_openai):
    chunks, embeddings = process_book("sample_book.txt", use_openai=use_openai)
    query = "What topics are mentioned in the book?"
    answer = rag_query(query, chunks, embeddings)
    assert "science" in answer.lower() and "history" in answer.lower() and "literature" in answer.lower()

# Добавим определение функции process_book, если она не существует в основном коде
def process_book(file_path: str, use_openai: bool = True):
    full_path = os.path.join(TEST_FILES_DIR, 'data', file_path)
    text_data = load_and_preprocess_text(full_path)
    chunks = split_into_chunks(text_data['text'], chunk_size=1000, overlap=100)
    embeddings = create_embeddings(chunks) if use_openai else [None] * len(chunks)
    return chunks, embeddings

@pytest.mark.parametrize("book_file", ["test_book.txt"])
def test_different_book_types(book_file, use_openai):
    chunks, embeddings = process_book(book_file, use_openai=use_openai)
    
    assert len(chunks) > 0
    assert len(embeddings) == len(chunks)
    if use_openai:
        assert all(len(embedding) == 1536 for embedding in embeddings)
    else:
        assert all(embedding is None for embedding in embeddings)

def test_rag_query(use_openai, mock_openai_service):
    chunks, embeddings = process_book("test_book.txt", use_openai=use_openai)
    query = "What is the main topic of the book?"
    book_data = BookDataInterface(chunks, embeddings, {})
    answer = rag_query(query, book_data, mock_openai_service)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Здесь можно добавить более конкретные проверки, если содержание test_book.txt известно
    # Сделал отдельный тест для проверки ответов на вопросы из test_ford_pinto_qa.py


    
    # Можно добавить более строгую проверку, например:
    # assert correct_count / total_count >= 0.3, f"Expected at least 30% correct answers, but got {correct_count}/{total_count}"

def test_pinecone_integration():
    mock_pinecone = MockPinecone(api_key="fake_key")
    mock_pinecone.set_error_probability(0.5)  # Увеличиваем вероятность ошибок для тестирования обработки ошибок
    
    pinecone_manager = PineconeManager(pinecone_client=mock_pinecone)
    
    # Test index creation with potential errors
    retries = 3
    while retries > 0:
        try:
            assert pinecone_manager.is_available()
            break
        except Exception:
            retries -= 1
    assert retries > 0, "Failed to create index after multiple attempts"
    
    # Test creating index that already exists
    another_manager = PineconeManager(pinecone_client=mock_pinecone)
    assert another_manager.is_available()
    
    chunks = ["Test chunk 1", "Test chunk 2"]
    embeddings = pinecone_manager.get_or_create_embeddings(chunks, lambda x: [[0.1] * 1536 for _ in x])
    assert len(embeddings) == 2
    assert all(len(emb) == 1536 for emb in embeddings)
    
    results = pinecone_manager.search_similar([0.1] * 1536, top_k=1)
    assert len(results) == 1
    assert "chunk" in results[0] and "score" in results[0]

    # Test error handling
    mock_pinecone.set_error_probability(1.0)  # Гарантируем ошибку
    with pytest.raises(Exception, match="Simulated Pinecone error"):
        pinecone_manager.search_similar([0.1] * 1536, top_k=1)
