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

def run_with_and_without_api(func):
    @functools.wraps(func)
    @pytest.mark.parametrize("use_api", [True, False])
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
