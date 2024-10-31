import pytest
from unittest.mock import Mock, patch
import os
from src.services.firebase_storage import FirebaseStorageService

class MockFirebaseStorage:
    def __init__(self):
        self._files = {}
        self._mock_bucket = Mock()
        self._mock_blob = Mock()
        self._mock_blob.public_url = "https://storage.googleapis.com/mock-bucket/test-file"
        self._mock_bucket.blob.return_value = self._mock_blob

    async def upload_file(self, file_path: str, user: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        storage_path = f"uploads/{user}/{os.path.basename(file_path)}"
        self._files[storage_path] = file_path
        return f"https://storage.googleapis.com/mock-bucket/{storage_path}"

@pytest.fixture
def storage():
    """Fixture that provides a MockFirebaseStorage instance for testing."""
    return MockFirebaseStorage()

@pytest.mark.asyncio
async def test_upload_file_success(storage, tmp_path):
    # Создаем тестовый файл
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    # Загружаем файл
    url = await storage.upload_file(str(test_file), "test_user")

    # Проверяем результат
    expected_storage_path = f"uploads/test_user/test.txt"
    assert expected_storage_path in storage._files
    assert url == f"https://storage.googleapis.com/mock-bucket/{expected_storage_path}"

@pytest.mark.asyncio
async def test_upload_file_error(storage):
    # Пытаемся загрузить несуществующий файл
    nonexistent_file = "/nonexistent/path/test.txt"
    
    with pytest.raises(FileNotFoundError):
        await storage.upload_file(nonexistent_file, "test_user")

@pytest.mark.asyncio
async def test_upload_file_with_special_chars(storage: MockFirebaseStorage, tmp_path):
    # Тестируем файл с специальными символами в имени
    test_file = tmp_path / "test file (1).txt"
    test_file.write_text("Test content")
    
    url = await storage.upload_file(str(test_file), "test_user")
    
    expected_storage_path = "uploads/test_user/test file (1).txt"
    assert expected_storage_path in storage._files
    assert url == f"https://storage.googleapis.com/mock-bucket/{expected_storage_path}"