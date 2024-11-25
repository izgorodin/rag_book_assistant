from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock, AsyncMock
import time
import random
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError, AsyncOpenAI
from src.book_data_factory import BookDataFactory
from src.utils.logger import get_main_logger
from src.vector_store_service import VectorStoreService
from tests.utils.error_factory import OpenAIErrorFactory
from tests.test_data.constants import (
    TEST_EMBEDDING_DIM,
    TEST_EMBEDDING_VALUES
)
from dataclasses import dataclass
import os
import firebase_admin
from src.interfaces.vector_store import VectorStore
import uuid

logger = get_main_logger()

class MockPinecone(VectorStore):
    """Mock implementation of VectorStore for testing."""
    
    def __init__(self):
        self._vectors = {}
        self._noise_vectors = {}
        self._is_initialized = False
        self._generate_noise_vectors()

    async def is_available(self) -> bool:
        """Асинхронная проверка доступности мока."""
        return self._is_initialized

    async def initialize(self) -> None:
        """Асинхронная инициализация мока."""
        self._is_initialized = True

    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Store vectors with metadata."""
        if not self._is_initialized:
            raise ValueError("Mock Pinecone index not initialized")
        for vector in vectors:
            vector_id = str(uuid.uuid4())
            self._vectors[vector_id] = {
                'values': vector['values'],
                'metadata': vector['metadata']
            }

    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self._is_initialized:
            raise ValueError("Mock Pinecone index not initialized")
            
        # Возвращаем список результатов в правильном формате
        results = []
        for vector_id, (values, metadata) in list(self._vectors.items())[:top_k]:
            results.append({
                'id': vector_id,
                'metadata': metadata,
                'score': 0.9  # Моковый score
            })
        
        # Есл нет результатов, добавляем тестовые данные
        if not results:
            results = [{
                'id': 'test_id',
                'metadata': {'text': 'test text'},
                'score': 0.9
            }]
            
        return results

    async def clear(self) -> None:
        """Clear all vectors."""
        self._vectors.clear()
        self._noise_vectors.clear()
        logger.debug("Cleared all vectors")

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict) -> bool:
        """Check if metadata matches filter conditions."""
        for key, value in filters.items():
            if key.startswith("$"):
                # Специальные операторы ($or, $and и т.д.)
                continue
            if metadata.get(key) != value:
                return False
        return True

    def _generate_noise_vectors(self, count: int = 100):
        """Генерирует шумовые векторы для более реалистичного тестирования"""
        base_texts = [
            "Случайный текст для тестирования",
            f"Событие произошло {random.randint(1, 28)} числа",
            f"В городе {random.choice(['Москва', 'Париж', 'Лондон'])} случилось что-то",
            f"Компания объявила о {random.choice(['слиянии', 'расширении', 'запуске'])}"
        ]
        
        for i in range(count):
            vector_id = f"noise_{i}"
            values = [random.uniform(-1, 1) for _ in range(1536)]
            text = f"{random.choice(base_texts)} #{i}"
            metadata = {"text": text}
            self._noise_vectors[vector_id] = (values, metadata)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Вычисляет косинусное сходство между векторам"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

class OpenAIConfig:
    """Конфигурация для OpenAI мока."""
    completion_response: str = "Test response"
    embedding_dimensions: int = 1536
    embedding_values: List[float] = None
    error_probability: float = 0.0
    error_message: str = "Test error"

class OpenAIMock:
    """Мок для OpenAI API с реалистичным поведением."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.chat = Mock()
        self.embeddings = Mock()
        self._setup_chat_completions()
        self._setup_embeddings()
        logger.debug(f"Created OpenAI mock with config: {config}")
    
    def _setup_chat_completions(self):
        """Настройка chat completions с обработкой ошибок."""
        def create(*args, **kwargs):
            if random.random() < self.config.error_probability:
                error_type = random.choice(MockFactory.ERROR_TYPES)
                raise OpenAIErrorFactory.create_error(
                    error_type, 
                    message=self.config.error_message
                )
            
            completion = Mock()
            message = Mock()
            message.content = self.config.completion_response
            completion.choices = [Mock(message=message)]
            return completion
            
        self.chat.completions.create.side_effect = create

    def _setup_embeddings(self):
        """Настройка embeddings с обработкой ошибок."""
        def create(*args, **kwargs):
            if random.random() < self.config.error_probability:
                error_type = random.choice(MockFactory.ERROR_TYPES)
                raise OpenAIErrorFactory.create_error(
                    error_type,
                    message=self.config.error_message
                )
            
            values = self.config.embedding_values or [0.1, 0.2]
            embedding = values * (self.config.embedding_dimensions // len(values))
            
            response = Mock()
            data = Mock()
            data.embedding = embedding
            response.data = [data]
            return response
            
        self.embeddings.create.side_effect = create

class MockFirebaseStorage:
    """Мок для Firebase Storage."""
    def __init__(self):
        self._files = {}

    async def upload_file(self, file_path: str, user_id: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        storage_path = f"uploads/{user_id}/{os.path.basename(file_path)}"
        self._files[storage_path] = file_path
        return f"https://storage.googleapis.com/mock-bucket/{storage_path}"

class MockBlob:
    def __init__(self, blob_path: str, files: Dict[str, str]):
        self.blob_path = blob_path
        self._files = files
        self.public_url = f"https://storage.googleapis.com/mock-bucket/{blob_path}"
        
    def upload_from_filename(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self._files[self.blob_path] = file_path
        
    def make_public(self):
        pass

class MockWebSocketManager:
    """Мок для WebSocket менеджера."""
    def __init__(self):
        self.messages = []
        self.connections = []
        logger.debug("Initialized MockWebSocketManager")

    async def connect(self, websocket: Mock):
        self.connections.append(websocket)
        
    async def disconnect(self, websocket: Mock):
        if websocket in self.connections:
            self.connections.remove(websocket)
            
    async def emit_progress(self, status: str = "", current: int = 0, total: int = 100):
        message = {
            "type": "progress",
            "status": status,
            "progress": (current / total) * 100 if total > 0 else 0,
            "current": current,
            "total": total
        }
        self.messages.append(message)
        for connection in self.connections:
            await connection.send_json(message)


class MockFactory:
    """Фабрика для создания тестовых моков."""
    
    ERROR_TYPES = [RateLimitError, APIError, APITimeoutError, APIConnectionError]
    
    @staticmethod
    def create_embedding_service(dim=1536, error_probability=0.0):
        """Creates a mock embedding service with async support."""
        mock_service = Mock()
        
        async def async_create_embeddings(texts: List[str]) -> List[List[float]]:
            """Async mock for creating embeddings."""
            if random.random() < error_probability:
                error_type = random.choice(MockFactory.ERROR_TYPES)
                raise OpenAIErrorFactory.create_error(error_type)
            
            # Create mock embeddings for each text
            return [[random.random() for _ in range(dim)] for _ in texts]
        
        # Set up the async method
        mock_service.create_embeddings = async_create_embeddings
        logger.debug(f"Created EmbeddingService mock (dim={dim})")
        return mock_service

    @staticmethod
    def create_openai_client() -> Mock:
        """Creates a mock AsyncOpenAI client."""
        mock_client = Mock(spec=AsyncOpenAI)  # Используем правильную спецификацию
        
        # Создаем структуру как у AsyncOpenAI
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        
        async def mock_create(**kwargs):
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="Test response"
                        )
                    )
                ]
            )
        
        # Используем AsyncMock для create
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        
        return mock_client

    @staticmethod
    def create_cache_mock(
        hit_rate: float = 0.8,
        embedding_dimensions: int = TEST_EMBEDDING_DIM,
        embedding_values: list = TEST_EMBEDDING_VALUES
    ) -> Mock:
        """
        Создает мок кэша.
        
        Возврщает один эмбеддинг:
        [0.1, 0.2]
        """
        cache = Mock()
        
        def get(*args, **kwargs):
            if random.random() < hit_rate:
                return embedding_values * (embedding_dimensions // len(embedding_values))
            return None
            
        cache.get.side_effect = get
        logger.debug(f"Created cache mock (dim={embedding_dimensions})")
        return cache

    @staticmethod
    def create_pinecone_client():
        return MockPinecone()  # Используем существующий MockPinecone

    @staticmethod
    def create_feature_extractor() -> Mock:
        """Создает мок для извлечения фич."""
        extractor = Mock()
        
        def extract_entities(text: str) -> List[str]:
            entities = []
            keywords = ["AI", "machine learning"]
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    entities.append(keyword)
            return entities
            
        def extract_dates(text: str) -> List[str]:
            return ["2023"] if "2023" in text else []
            
        extractor.extract_entities.side_effect = extract_entities
        extractor.extract_dates.side_effect = extract_dates
        return extractor

    @staticmethod
    def create_firebase_storage():
        return MockFirebaseStorage()

    @staticmethod
    def create_websocket_manager() -> MockWebSocketManager:
        """Создает мок WebSocket менеджера."""
        return MockWebSocketManager()

    @staticmethod
    def create_websocket() -> Mock:
        """Создает мок WebSocket соединения."""
        websocket = Mock()
        websocket.send_json = Mock()
        websocket.receive_text = Mock(return_value="test message")
        return websocket

    @staticmethod
    def create_book_factory() -> BookDataFactory:
        vector_store = MockFactory.create_pinecone_client()
        embedding_service = MockFactory.create_embedding_service()
        vector_store_service = VectorStoreService(vector_store, embedding_service)
        return BookDataFactory(vector_store_service, embedding_service)

