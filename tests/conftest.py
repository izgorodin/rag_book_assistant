import pytest
from src.vector_store_service import VectorStoreService
from src.cache_manager import CacheManager
from src.openai_service import OpenAIService
from tests.utils.mock_factory import MockFactory
from unittest.mock import MagicMock

# OpenAI Test Configuration
TEST_EMBEDDING_DIM = 1536
TEST_BATCH_SIZE = 5
TEST_EMBEDDING_VALUES = [[0.1] * TEST_EMBEDDING_DIM]
TEST_COMPLETION_RESPONSE = "Test response"
TEST_ERROR_PROBABILITY = 0.0
TEST_ERROR_MESSAGE = "Test error"

# Test Data
TEST_CHUNKS = [
    "В 1999 году произошло важное событие",
    "Технология развивается быстро",
    "Новое событие в 2024 году",
    "Искусственный интеллект меняет мир"
]

TEST_QUERIES = [
    "Когда была основана Google?",
    "Что такое BackRub?",
    "Расскажи про Chrome",
    "Какие события запланированы?"
]

# Test Processing Parameters
TEST_CHUNK_SIZE = 1000
TEST_CHUNK_OVERLAP = 150

@pytest.fixture
def mock_cache_manager():
    return MagicMock(spec=CacheManager)

@pytest.fixture
def mock_openai_service():
    return MagicMock(spec=OpenAIService)

@pytest.fixture
async def vector_store():
    store = MockFactory.create_pinecone_client()
    await store.initialize()
    yield store
    await store.clear()

@pytest.fixture
async def vector_service(vector_store):
    embedding_service = MockFactory.create_embedding_service()
    service = VectorStoreService(vector_store, embedding_service)
    await service.initialize()
    return service

@pytest.fixture
async def test_book_data():
    text = """
    В 1999 году Сергей Брин и Ларри Пейдж основали компанию Google в Калифорнии.
    Изначально поисковик назывался BackRub и работал на серверах Стэнфорда.
    15 сентября 2008 года была выпущена первая версия браузера Chrome.
    Искусственный интеллект и машинное обучение трансформируют индустрии.
    15 марта 2024 года состоится важная конференция по ИИ.
    """
    book_factory = MockFactory.create_book_factory()
    return await book_factory.create_from_text(text)

@pytest.fixture
def test_embeddings():
    """Fixture for test embeddings"""
    return [[0.1] * TEST_EMBEDDING_DIM for _ in range(len(TEST_CHUNKS))]
