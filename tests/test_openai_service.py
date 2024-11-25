import pytest
from src.openai_service import OpenAIService
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
from tests.conftest import TEST_CHUNKS, TEST_EMBEDDING_DIM, TEST_COMPLETION_RESPONSE
from tests.utils.mock_factory import MockFactory

@pytest.fixture
def mock_openai_client():
    """Фикстура для мок клиента OpenAI."""
    return MockFactory.create_openai_client()

@pytest.fixture
def openai_service(mock_openai_client):
    """Фикстура для сервиса OpenAI с мок клиентом."""
    service = OpenAIService(client=mock_openai_client)
    return service

@pytest.mark.asyncio
async def test_generate_answer(openai_service):
    """Test answer generation with mocked OpenAI client."""
    # Arrange
    mock_client = MockFactory.create_openai_client()
    openai_service.client = mock_client

    # Act
    answer = await openai_service.generate_answer("Test query", "Test context")

    # Assert
    assert answer == "Test response"
    
    # Verify the mock was called with correct parameters
    assert mock_client.chat.completions.create.await_count == 1
    mock_client.chat.completions.create.assert_awaited_once_with(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system", 
                "content": "You are an AI assistant specialized in accurately extracting information from the provided text. Use only the information given in the context and avoid adding any external data. If the required information is not available in the context, briefly explain what is known and mention that the specific information is missing."
            },
            {
                "role": "user", 
                "content": "Context:\nTest context\n\nQuestion:\nTest query\n\nPlease provide an answer based on the above context."
            }
        ],
        max_tokens=15000
    )

@pytest.mark.asyncio
async def test_create_embeddings(openai_service):
    """Test embeddings creation."""
    # Arrange
    mock_embedding_service = MockFactory.create_embedding_service()
    openai_service.set_embedding_service(mock_embedding_service)
    
    # Act
    embeddings = await openai_service.create_embeddings(["Test text"])
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1  # One text = one embedding
    assert len(embeddings[0]) == 1536  # Check dimension
    assert all(isinstance(x, float) for x in embeddings[0])  # Check all values are floats

@pytest.mark.asyncio
async def test_error_simulation():
    """Test API error simulation."""
    service = OpenAIService()
    mock_embedding_service = MockFactory.create_embedding_service(error_probability=1.0)
    service.set_embedding_service(mock_embedding_service)
    
    with pytest.raises((RateLimitError, APIError, APITimeoutError, APIConnectionError)):
        await service.create_embeddings(["Test"])

@pytest.mark.asyncio
async def test_generate_answer_with_error(openai_service):
    """Test error handling in answer generation."""
    mock_client = MockFactory.create_openai_client(error_probability=1.0)
    openai_service.client = mock_client
    
    response = await openai_service.generate_answer("Test query", "Test context")
    assert "Sorry, I encountered an error" in response

@pytest.mark.asyncio
async def test_embedding_service_not_initialized():
    """Test uninitialized embedding service error."""
    service = OpenAIService()
    with pytest.raises(ValueError):
        await service.create_embeddings(["Test"])

def test_set_embedding_service():
    """Тест установки embedding_service."""
    service = OpenAIService()
    mock_embedding_service = MockFactory.create_embedding_service()
    service.set_embedding_service(mock_embedding_service)
    assert service.embedding_service == mock_embedding_service