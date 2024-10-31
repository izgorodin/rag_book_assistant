from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock
import time
import random
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
from src.utils.logger import get_main_logger
from tests.utils.error_factory import OpenAIErrorFactory
from tests.test_data.constants import (
    TEST_EMBEDDING_DIM,
    TEST_EMBEDDING_VALUES
)
from dataclasses import dataclass
import os
import firebase_admin

logger = get_main_logger()

class MockPineconeIndex:
    """Мок-класс для Pinecone индекса."""
    def __init__(self):
        self.vectors = {}
        logger.debug("Initialized MockPineconeIndex")

    def upsert(self, vectors: List[tuple]):
        for id, embedding, metadata in vectors:
            self.vectors[id] = (embedding, metadata)
        logger.debug(f"Upserted {len(vectors)} vectors")

    def query(self, vector: List[float], top_k: int = 1, include_metadata: bool = True) -> Dict[str, Any]:
        matches = [{"id": id, "score": 0.9, "metadata": metadata} 
                   for id, (_, metadata) in list(self.vectors.items())[:top_k]]
        logger.debug(f"Query returned {len(matches)} matches")
        return {"matches": matches}

    def delete(self, delete_all: bool = False):
        if delete_all:
            self.vectors.clear()
            logger.debug("Cleared all vectors")

class MockPinecone:
    """Мок для Pinecone хранилища векторов."""
    
    def __init__(self):
        self._vectors: Dict[str, Tuple[List[float], Dict]] = {}
        self._error_probability: float = 0.0
        logger.debug("Initialized MockPinecone")

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Сохраняет векторы в моке.
        
        Args:
            vectors: Список словарей с полями id, values, metadata
        """
        for vector in vectors:
            vector_id = vector['id']
            values = vector['values']
            metadata = vector['metadata']
            self._vectors[vector_id] = (values, metadata)
        logger.debug(f"Stored {len(vectors)} vectors")

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Ищет похожие векторы.
        
        Args:
            vector: Вектор для поиска
            top_k: Количество результатов
            include_metadata: Включать ли метаданные
        
        Returns:
            Dict с полем matches, содержащим список найденных векторов
        """
        matches = []
        for vector_id, (values, metadata) in self._vectors.items():
            match = {
                'id': vector_id,
                'score': 0.9  # Фиксированный скор для тестов
            }
            if include_metadata:
                match['metadata'] = metadata
            matches.append(match)
            
        return {'matches': matches[:top_k]}

    def delete(self, delete_all: bool = False):
        if delete_all:
            self._vectors.clear()
            logger.debug("Cleared all vectors")

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
    def create_embedding_service(
        error_probability: float = 0.0,
        embedding_dimensions: int = TEST_EMBEDDING_DIM,
        embedding_values: List[float] = TEST_EMBEDDING_VALUES
    ) -> Mock:
        """Создает мок сервиса эмбеддингов."""
        service = Mock()
        
        def create_embeddings(texts: List[str]) -> List[List[float]]:
            if random.random() < error_probability:
                error_type = random.choice(MockFactory.ERROR_TYPES)
                raise OpenAIErrorFactory.create_error(error_type)
            
            # Создаем эмбеддинг для каждого текста
            base_embedding = embedding_values * (embedding_dimensions // len(embedding_values))
            return [base_embedding.copy() for _ in texts]  # Важно: copy() для каждого эмбеддинга
            
        service.create_embeddings.side_effect = create_embeddings
        logger.debug(f"Created EmbeddingService mock (dim={embedding_dimensions})")
        return service

    @staticmethod
    def create_openai_client(
        completion_response: str = "Test response",
        error_probability: float = 0.0
    ) -> Mock:
        """
        Создает мок OpenAI клиента.
        
        Args:
            completion_response: Ответ для chat completion
            error_probability: Вероятность ошибки
            
        Returns:
            Mock OpenAI клиента
        """
        client = Mock()
        
        # Настройка chat completion
        completion = Mock()
        message = Mock()
        message.content = completion_response
        completion.choices = [Mock(message=message)]
        
        def create_chat_completion(*args, **kwargs):
            if random.random() < error_probability:
                error_type = random.choice(MockFactory.ERROR_TYPES)
                raise OpenAIErrorFactory.create_error(
                    error_type,
                    message=completion_response
                )
            return completion
        
        client.chat.completions.create.side_effect = create_chat_completion
        
        return client

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

