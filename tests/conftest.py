import pytest
from unittest.mock import Mock, patch
from openai import OpenAI
import functools
import os

def pytest_addoption(parser):
    parser.addoption(
        "--use-real-api",
        action="store_true",
        default=False,
        help="run tests with real API calls"
    )

@pytest.fixture(scope="session")
def use_real_api(request):
    return request.config.getoption("--use-real-api")

@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_client.embeddings = Mock()
    mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content="Mocked response"))])
    return mock_client

@pytest.fixture
def patch_openai(mock_openai_client, use_real_api):
    if use_real_api:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
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
    @pytest.mark.parametrize("use_api", [True, False])
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper