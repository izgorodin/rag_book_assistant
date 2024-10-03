import pytest
from unittest.mock import Mock, patch
from openai import OpenAI
import functools


@pytest.fixture
def mock_openai_client():
    mock_client = Mock(spec=OpenAI)
    mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])
    mock_client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content="Mocked response"))])
    return mock_client

@pytest.fixture
def patch_openai(mock_openai_client):
    with patch('openai.OpenAI', return_value=mock_openai_client):
        yield mock_openai_client

@pytest.fixture
def sample_text():
    return "This is a sample text for testing purposes. It contains multiple sentences and should be long enough for chunking."

@pytest.fixture
def sample_chunks():
    return [
        "This is chunk one for testing.",
        "This is chunk two for testing.",
        "This is chunk three for testing."
    ]

@pytest.fixture
def sample_embeddings():
    return [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

def run_with_and_without_api(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Получаем все фикстуры, которые нужны для теста
        fixtures = pytest.fixtures(func)
        
        # Запускаем тест с мокированным API
        with patch('openai.OpenAI'):
            fixture_values = {name: pytest.fixture()(request=None) for name in fixtures}
            func(**fixture_values, use_api=False)
        
        # Запускаем тест с реальным API, если установлен OPENAI_API_KEY
        import os
        if os.getenv("OPENAI_API_KEY"):
            fixture_values = {name: pytest.fixture()(request=None) for name in fixtures}
            func(**fixture_values, use_api=True)
        else:
            pytest.skip("Skipping API test because OPENAI_API_KEY is not set")
    
    return wrapper