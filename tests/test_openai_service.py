import pytest
from unittest.mock import patch, MagicMock
from src.openai_service import OpenAIService
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError

@pytest.fixture
def openai_service():
    return OpenAIService(api_key="test_key")

def test_generate_answer(openai_service):
    with patch.object(openai_service.client.chat.completions, 'create') as mock_create:
        mock_create.return_value.choices[0].message.content = "Test answer"
        answer = openai_service.generate_answer("Test query", "Test context")
        assert answer == "Test answer"

def test_create_embeddings(openai_service):
    with patch.object(openai_service.client.embeddings, 'create') as mock_create:
        mock_create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        embeddings = openai_service.create_embeddings(["Test text"])
        assert embeddings == [[0.1, 0.2, 0.3]]

@pytest.mark.parametrize("error_type, error_message", [
    (RateLimitError, "Rate limit exceeded"),
    (APIError, "API error"),
    (APITimeoutError, "Request timed out"),
    (APIConnectionError, "Connection error")
])
def test_handle_openai_error(openai_service, error_type, error_message):
    error = OpenAIService.create_test_exception(error_type, error_message)
    handled_error = openai_service._handle_openai_error(error)
    assert isinstance(handled_error, error_type)
    assert error_message in str(handled_error)

def test_create_test_exception():
    for error_type in [RateLimitError, APIError, APITimeoutError, APIConnectionError]:
        error = OpenAIService.create_test_exception(error_type, "Test error")
        assert isinstance(error, error_type)
        if error_type != APITimeoutError:
            assert "Test error" in str(error)
        else:
            assert "Request timed out" in str(error)
