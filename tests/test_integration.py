import pytest
import os
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import create_embeddings
from src.rag import rag_query
import logging
from tests.ford_pinto_qa_data import qa_pairs
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    full_path = os.path.join(TEST_FILES_DIR, file_path)
    text_data = load_and_preprocess_text(full_path)
    chunks = split_into_chunks(text_data, chunk_size=1000, overlap=100)
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

def test_rag_query(use_openai):
    chunks, embeddings = process_book("test_book.txt", use_openai=use_openai)
    query = "What is the main topic of the book?"
    answer = rag_query(query, chunks, embeddings)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Здесь можно добавить более конкретные проверки, если содержание test_book.txt известно

def test_ford_pinto_rag_queries(use_openai):
    file_path = os.path.join(TEST_FILES_DIR, "ford.txt")
    assert os.path.exists(file_path), f"Test file not found: {file_path}"
    logger.debug(f"Starting Ford Pinto RAG queries test with use_openai={use_openai}")
    chunks, embeddings = process_book("ford.txt", use_openai=use_openai)
    
    logger.debug(f"Number of chunks: {len(chunks)}")
    logger.debug(f"Number of embeddings: {len(embeddings)}")
    logger.debug(f"Type of first embedding: {type(embeddings[0]) if embeddings else 'No embeddings'}")
    
    results = []
    
    for qa_pair in qa_pairs:
        query = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        logger.debug(f"Processing question: {query}")
        try:
            answer = rag_query(query, chunks, embeddings)
        except Exception as e:
            logger.error(f"Error in rag_query: {str(e)}")
            logger.error(traceback.format_exc())
            answer = f"Error: {str(e)}"
        
        is_correct = isinstance(answer, str) and len(answer) > 0 and not answer.startswith("Error:")
        if is_correct:
            is_correct = any(word.lower() in answer.lower() for word in expected_answer.split() if len(word) > 3)
        
        result = {
            "question": query,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "is_correct": is_correct
        }
        results.append(result)
        
        logger.debug(f"Question: {query}")
        logger.debug(f"Expected answer: {expected_answer}")
        logger.debug(f"Actual answer: {answer}")
        logger.debug(f"Is correct: {is_correct}")
        
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    
    logger.info(f"Test completed. Correct answers: {correct_count}/{total_count}")
    
    for result in results:
        if not result["is_correct"]:
            logger.warning(f"Incorrect answer for question: {result['question']}")
            logger.warning(f"Expected: {result['expected_answer']}")
            logger.warning(f"Got: {result['actual_answer']}")
    
    assert correct_count > 0, f"Expected at least one correct answer, but got {correct_count}/{total_count}"
    
    # Можно добавить более строгую проверку, например:
    # assert correct_count / total_count >= 0.3, f"Expected at least 30% correct answers, but got {correct_count}/{total_count}"
