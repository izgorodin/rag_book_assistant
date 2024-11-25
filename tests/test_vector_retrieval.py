import pytest
from src.vector_store_service import VectorStoreService
from tests.conftest import TEST_EMBEDDING_DIM
from tests.utils.mock_factory import MockFactory
import os

@pytest.fixture
async def vector_store():
    store = MockFactory.create_pinecone_client()
    await store.initialize()
    yield store
    await store.clear()

@pytest.fixture
async def vector_service(vector_store):
    service = VectorStoreService(vector_store)
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

@pytest.mark.asyncio
class TestVectorRetrieval:
    """Test suite for vector retrieval capabilities"""
    
    @pytest.fixture(autouse=True)
    async def setup(self, vector_service):
        await vector_service.initialize()

    async def test_basic_retrieval(self, vector_service):
        """Test basic vector storage and retrieval"""
        chunks = ["test chunk 1", "test chunk 2"]
        embeddings = [[0.1] * TEST_EMBEDDING_DIM for _ in range(2)]
        vectors = [
            {'values': emb, 'metadata': {'text': chunk}} 
            for emb, chunk in zip(embeddings, chunks)
        ]
        
        await vector_service.store_vectors(vectors)
        query_vector = [0.1] * TEST_EMBEDDING_DIM
        results = await vector_service.search_vectors(query_vector)
        
        assert len(results) > 0
        assert 'metadata' in results[0]
        assert 'text' in results[0]['metadata']

    async def test_factual_retrieval_accuracy(self, test_book_data):
        """Test accuracy of fact retrieval"""
        book_data = await test_book_data
        test_cases = [
            {
                "query": "Когда была основана Google?",
                "expected_facts": ["1999", "Сергей Брин", "Ларри Пейдж"],
                "unexpected_facts": ["2008", "Chrome"]
            }
        ]
        
        for case in test_cases:
            chunks = await book_data.get_relevant_chunks(case["query"])
            # Проверяем наличие ожидаемых фактов
            for fact in case["expected_facts"]:
                assert any(fact in chunk for chunk in chunks), \
                    f"Ожидаемый факт '{fact}' не найден для запроса '{case['query']}'"
            
            # Проверяем отсутствие не ожидаемых фактов
            for fact in case["unexpected_facts"]:
                assert not any(fact in chunk for chunk in chunks), \
                    f"Неожидаемы факт '{fact}' найден для запроса '{case['query']}'"

    async def test_retrieval_quality_metrics(self, test_book_data):
        """Тест метрик качества поиска"""
        book_data = await test_book_data
        test_cases = [
            {
                "query": "Когда была основана Google?",
                "relevant_chunks": ["В 1999 году Сергей Брин и Ларри Пейдж основали компанию Google"],
                "irrelevant_chunks": ["15 сентября 2008 года была выпущена первая версия браузера Chrome"]
            }
        ]
        
        for case in test_cases:
            chunks = await book_data.get_relevant_chunks(case["query"])
            precision = len(set(chunks) & set(case["relevant_chunks"])) / len(chunks)
            recall = len(set(chunks) & set(case["relevant_chunks"])) / len(case["relevant_chunks"])
            
            assert precision >= 0.5, f"Низкая точность ({precision}) для запроса '{case['query']}'"
            assert recall >= 0.5, f"Низкий recall ({recall}) для запроса '{case['query']}'"