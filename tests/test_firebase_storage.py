import pytest
import os
from tests.utils.mock_factory import MockFactory

@pytest.fixture
def storage():
    return MockFactory.create_firebase_storage()

@pytest.mark.asyncio
async def test_upload_file_success(storage, tmp_path):
    # Создаем тестовый файл
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Загружаем файл
    url = await storage.upload_file(str(test_file), "test_user")
    
    # Проверяем результат
    expected_storage_path = "uploads/test_user/test.txt"
    assert expected_storage_path in storage._files
    assert storage._files[expected_storage_path] == str(test_file)
    assert url == f"https://storage.firebase.com/{expected_storage_path}"

@pytest.mark.asyncio
async def test_upload_file_error(storage):
    # Пытаемся загрузить несуществующий файл
    nonexistent_file = "/nonexistent/path/test.txt"
    
    with pytest.raises(FileNotFoundError):
        await storage.upload_file(nonexistent_file, "test_user")

@pytest.mark.asyncio
async def test_upload_file_with_special_chars(storage, tmp_path):
    # Тестируем файл с специальными символами в имени
    test_file = tmp_path / "test file (1).txt"
    test_file.write_text("Test content")
    
    url = await storage.upload_file(str(test_file), "test_user")
    
    expected_storage_path = "uploads/test_user/test file (1).txt"
    assert expected_storage_path in storage._files