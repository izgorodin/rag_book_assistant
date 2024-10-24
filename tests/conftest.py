import pytest
from unittest.mock import Mock, patch
from openai import OpenAI
import os
import time
from src.logger import setup_logger

logger = setup_logger()

def pytest_addoption(parser):
    parser.addoption(
        "--use-real-api",
        action="store_true",
        default=False,
        help="run tests with real API calls"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    ) 

@pytest.fixture(params=[True, False])
def use_openai(request):
    return request.param

@pytest.fixture(scope="session")
def use_real_api(request):
    return request.config.getoption("--use-real-api")

@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])
    mock_client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content="Mocked response about fiction with wizards and magic"))])
    return mock_client

@pytest.fixture
def openai_client(use_real_api, mock_openai_client):
    if use_real_api:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        return mock_openai_client

@pytest.fixture
def patch_openai(use_real_api, mock_openai_client):
    if use_real_api:
        yield
    else:
        with patch('openai.OpenAI', return_value=mock_openai_client):
            yield mock_openai_client

@pytest.fixture(autouse=True)
def timer(request):
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"Test {request.node.name} took {end_time - start_time:.2f} seconds")

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
