"""Константы для тестов."""

# Эмбеддинги
TEST_EMBEDDING_DIM = 1536  # Размерность эмбеддингов OpenAI
TEST_EMBEDDING_VALUES = [0.1, 0.2]  # Базовые значения для эмбеддингов

# Тексты
TEST_TEXTS = [
    "This is a sample text for testing purposes.",
    "Another sample text for testing.",
    "Third sample text for testing."
]

# API ответы
TEST_API_RESPONSES = {
    "completion": "Test response",
    "error": "An error occurred during processing"
}

# Параметры чанкинга
TEST_CHUNK_SIZE = 1000
TEST_CHUNK_OVERLAP = 150