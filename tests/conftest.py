import pytest
from unittest.mock import Mock, patch
from openai import OpenAI
import os
import time
from src.openai_service import OpenAIService
from src.utils.logger import get_main_logger
from tests.utils.mock_factory import MockFactory, MockFirebaseStorage
from tests.test_data.constants import (
    TEST_EMBEDDING_DIM,
    TEST_EMBEDDING_VALUES,
    TEST_TEXTS,
    TEST_API_RESPONSES
)
import sys
import asyncio
from fastapi.testclient import TestClient
from src.web.app import app, get_storage_service

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_main_logger()


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
    """Фикстура для мок клиента OpenAI."""
    return MockFactory.create_openai_client()

@pytest.fixture
def mock_embedding_service():
    """Фикстура для мок сервиса эмбеддингов."""
    return MockFactory.create_embedding_service(
        embedding_dimensions=TEST_EMBEDDING_DIM,
        embedding_values=TEST_EMBEDDING_VALUES
    )

@pytest.fixture
def mock_cache():
    """Фикстура для мок кэша с тестовыми эмбеддингами."""
    return MockFactory.create_cache_mock(
        embedding_dimensions=TEST_EMBEDDING_DIM
    )

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

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def storage_service():
    mock_storage = MockFirebaseStorage()
    # Важно: переопределяем глобальный storage_service
    app.dependency_overrides[get_storage_service] = lambda: mock_storage
    return mock_storage

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Мок для OpenAI в тестах"""
    if os.getenv('TESTING') == 'true':
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Mocked response"))]
        )
        monkeypatch.setattr('openai.OpenAI', lambda **kwargs: mock_client)
        monkeypatch.setattr(OpenAIService, '_get_client', lambda self: mock_client)

@pytest.fixture(autouse=True)
def mock_firebase(monkeypatch):
    """Мок для Firebase в тестах"""
    if os.getenv('TESTING') == 'true':
        mock_storage = Mock()
        mock_storage.child.return_value = mock_storage
        mock_storage.put.return_value = None
        mock_storage.get_url.return_value = "http://mock-url"
        monkeypatch.setattr('pyrebase.initialize_app', lambda config: Mock(storage=lambda: mock_storage))
