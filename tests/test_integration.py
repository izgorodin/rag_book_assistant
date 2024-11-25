import pytest
from unittest.mock import MagicMock
from src.embedding import EmbeddingService
from src.rag import rag_query
from src.book_data_interface import BookDataInterface
from src.book_data_factory import BookDataFactory
from src.vector_store_service import VectorStoreService
from src.cache_manager import CacheManager
from src.openai_service import OpenAIService
from tests import conftest
from tests.utils.mock_factory import MockFactory

@pytest.fixture
def mock_services():
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.create_embeddings.return_value = [[0.1] * conftest.TEST_EMBEDDING_DIM]
    
    openai_service = MagicMock(spec=OpenAIService)
    openai_service.generate_answer.return_value = "Test response"
    
    cache_manager = MagicMock(spec=CacheManager)
    cache_manager.get.return_value = None
    
    mock_pinecone = MockFactory.create_pinecone_client()
    vector_store_service = VectorStoreService(mock_pinecone)
    
    return embedding_service, vector_store_service, openai_service, cache_manager

@pytest.fixture
def book_factory(mock_services):
    embedding_service, vector_store_service, _, _ = mock_services
    return BookDataFactory(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service
    )

def test_book_processing(book_factory, tmp_path):
    """Тест обработки книги с проверкой основных свойств"""
    test_book = tmp_path / "test_book.txt"
    test_book.write_text("""
    В 1999 году компания Google была основана в Калифорнии.
    Искусственный интеллект и машинное обучение трансформируют индустрии.
    15 марта 2024 года состоится важная конференция по ИИ.
    """)
    
    with open(test_book, 'r') as f:
        book_data = book_factory.create_from_text(f.read())
    
    # Проверяем базовые свойства
    assert len(book_data.get_chunks()) > 0
    assert len(book_data.get_embeddings()) == len(book_data.get_chunks())
    assert all(len(emb) == 1536 for emb in book_data.get_embeddings())
    
    # Проверяем извлечение фактов
    assert isinstance(book_data.get_dates(), list)
    assert isinstance(book_data.get_entities(), list)
    assert isinstance(book_data.get_key_phrases(), list)
    assert any("1999" in date for date in book_data.get_dates())
    assert any("2024" in date for date in book_data.get_dates())
    assert any("Google" in str(entity) for entity in book_data.get_entities())
    assert any("ИИ" in phrase for phrase in book_data.get_key_phrases())

@pytest.mark.asyncio
async def test_rag_query_integration(book_factory):
    """Тест интеграции RAG запроса"""
    test_text = """
    В 1999 году компания Google была основана в Калифорнии.
    Искусственный интеллект и машинное обучение трансформируют индустрии.
    15 марта 2024 года состоится важная конференция по ИИ.
    """
    book_data = await book_factory.create_from_text(test_text)
    
    embedding_service = MockFactory.create_embedding_service()
    openai_service = MockFactory.create_openai_client()
    
    # Тестируем разные типы запросов
    queries = [
        "Когда была основана Google?",
        "Что трансформирует индустрии?",
        "Какие события запланированы на 2024 год?"
    ]
    
    for query in queries:
        response = await rag_query(
            query=query,
            book_data=book_data,
            openai_service=openai_service,
            embedding_service=embedding_service
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Проверяем, что ответ содержит релевантную информацию
        if "Google" in query:
            assert "1999" in response
        if "трансформирует" in query:
            assert "интеллект" in response.lower() or "ии" in response.lower()
        if "2024" in query:
            assert "конференция" in response.lower()
