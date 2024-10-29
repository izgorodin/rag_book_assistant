import pytest
from src.openai_service import OpenAIService
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
from tests.utils.mock_factory import MockFactory
from tests.utils.error_factory import OpenAIErrorFactory

@pytest.fixture
def mock_openai_client():
    """Фикстура для мок клиента OpenAI."""
    return MockFactory.create_openai_client()

@pytest.fixture
def openai_service(mock_openai_client):
    """Фикстура для сервиса OpenAI с мок клиентом."""
    service = OpenAIService(api_key="test_key")
    service.client = mock_openai_client
    return service

def test_generate_answer(openai_service):
    """Тест генерации ответа."""
    answer = openai_service.generate_answer("Test query", "Test context")
    assert answer == "Test response"
    openai_service.client.chat.completions.create.assert_called_once()

def test_create_embeddings(openai_service):
    """Тест создания эмбеддингов."""
    mock_embedding_service = MockFactory.create_embedding_service()
    openai_service.set_embedding_service(mock_embedding_service)
    
    embeddings = openai_service.create_embeddings(["Test text"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1536
    mock_embedding_service.create_embeddings.assert_called_once()

@pytest.mark.parametrize("error_type,error_message", [
    (RateLimitError, "Rate limit exceeded"),
    (APIError, "API error"),
    (APITimeoutError, "Request timed out"),
    (APIConnectionError, "Connection error")
])
def test_handle_openai_error(openai_service, error_type, error_message):
    """Тест обработки различных ошибок OpenAI."""
    error = openai_service.create_test_exception(error_type, error_message)
    handled_error = openai_service._handle_openai_error(error)
    assert isinstance(handled_error, error_type)
    assert str(handled_error) != ""

def test_openai_connectivity():
    """Тест прямого подключения к OpenAI API."""
    service = OpenAIService()
    mock_embedding_service = MockFactory.create_embedding_service()
    service.set_embedding_service(mock_embedding_service)
    
    try:
        result = service.create_embeddings(["Test connection"])
        assert len(result) == 1
        assert len(result[0]) == 1536
    except Exception as e:
        pytest.fail(f"OpenAI connection failed: {str(e)}")

def test_error_simulation():
    """Тест симуляции ошибок API."""
    service = OpenAIService()
    mock_embedding_service = MockFactory.create_embedding_service(error_probability=1.0)
    service.set_embedding_service(mock_embedding_service)
    
    with pytest.raises((RateLimitError, APIError, APITimeoutError, APIConnectionError)) as exc_info:
        service.create_embeddings(["Test"])
    assert isinstance(exc_info.value, (RateLimitError, APIError, APITimeoutError, APIConnectionError))

def test_generate_answer_with_error(openai_service):
    """Тест обработки ошибок при генерации ответа."""
    mock_client = MockFactory.create_openai_client(error_probability=1.0)
    openai_service.client = mock_client
    
    response = openai_service.generate_answer("Test query", "Test context")
    assert "Sorry, I encountered an error" in response

def test_embedding_service_not_initialized():
    """Тест проверки ошибки неинициализированного embedding_service."""
    service = OpenAIService()
    with pytest.raises(ValueError) as exc_info:
        service.create_embeddings(["Test"])
    assert "EmbeddingService not initialized" in str(exc_info.value)

def test_set_embedding_service():
    """Тест установки embedding_service."""
    service = OpenAIService()
    mock_embedding_service = MockFactory.create_embedding_service()
    service.set_embedding_service(mock_embedding_service)
    assert service.embedding_service == mock_embedding_service
