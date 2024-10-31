import pytest
from fastapi.testclient import TestClient
from src.web.app import app
import os
import base64
from src.config import UPLOAD_FOLDER
from src.book_data_interface import BookDataInterface
from unittest.mock import Mock, patch
from src.services.firebase_storage import FirebaseStorageService

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers():
    credentials = base64.b64encode(b"admin:admin1q2w3e").decode()
    return {"Authorization": f"Basic {credentials}"}

@pytest.fixture
def test_file():
    """Создает временный тестовый файл"""
    content = "Test book content"
    filename = "test.txt"
    with open(filename, "w") as f:
        f.write(content)
    yield filename
    # Очистка
    if os.path.exists(filename):
        os.remove(filename)

@pytest.fixture
def mock_firebase():
    with patch('firebase_admin.credentials.Certificate') as mock_cert, \
         patch('firebase_admin.initialize_app') as mock_init, \
         patch('firebase_admin.storage.bucket') as mock_bucket:
        
        # Настраиваем мок для blob
        mock_blob = Mock()
        mock_blob.public_url = "https://storage.googleapis.com/test-bucket/test-file.txt"
        
        # Настраиваем мок для bucket
        bucket_mock = Mock()
        bucket_mock.blob.return_value = mock_blob
        mock_bucket.return_value = bucket_mock
        
        yield {
            'cert': mock_cert,
            'init': mock_init,
            'bucket': mock_bucket,
            'blob': mock_blob
        }

@pytest.fixture
def storage_service(mock_firebase):
    return FirebaseStorageService()

def test_auth_required(client):
    """Проверка что без авторизации доступ запрещен"""
    response = client.post("/upload")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_file_upload_success(client, auth_headers, test_file, mock_firebase):
    """Проверка успешной загрузки файла"""
    with open(test_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": (test_file, f, "text/plain")},
            headers=auth_headers
        )
    
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert "storage_url" in result
    assert result["storage_url"].startswith("https://storage.googleapis.com/")

@pytest.mark.asyncio
async def test_file_upload_firebase_error(client, auth_headers, test_file, storage_service):
    # Устанавливаем ошибку для следующей загрузки
    storage_service.set_upload_error(Exception("Firebase error"))
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": (os.path.basename(test_file), f, "text/plain")},
            headers=auth_headers
        )
    
    assert response.status_code == 500
    assert "Firebase upload error" in response.json()["detail"]
    
    # Проверяем, что временный файл был удален
    uploaded_file = os.path.join(UPLOAD_FOLDER, os.path.basename(test_file))
    assert not os.path.exists(uploaded_file)

def test_invalid_file_type(client, auth_headers):
    """Проверка отклонения неподдерживаемого типа файла"""
    content = b"Invalid file content"
    response = client.post(
        "/upload",
        files={"file": ("test.exe", content, "application/x-msdownload")},
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_book_data_persistence(client, auth_headers, test_file):
    """Проверка сохранения и восстановления данных книги"""
    # Загружаем файл
    with open(test_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": (test_file, f, "text/plain")},
            headers=auth_headers
        )
    assert response.status_code == 200
    
    # Проверяем что данные книги доступны
    response = client.get("/check_book_loaded", headers=auth_headers)
    assert response.json()["book_loaded"] is True
    
    # Проверяем возможность задать вопрос
    response = client.get(
        "/ask?question=test question",
        headers=auth_headers
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_websocket_progress(client, auth_headers, test_file):
    """Проверка WebSocket уведомлений о прогрессе"""
    with client.websocket_connect("/ws") as websocket:
        # Загружаем файл
        with open(test_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": (test_file, f, "text/plain")},
                headers=auth_headers
            )
        
        # Получаем сообщения о прогрессе
        data = websocket.receive_json()
        assert "progress" in data
        assert "status" in data

@pytest.mark.asyncio
async def test_cleanup(client, auth_headers, test_file):
    temp_files = []
    try:
        # Создаем временные файлы
        temp_file = os.path.join(UPLOAD_FOLDER, "temp_test.txt")
        temp_files.append(temp_file)
        with open(temp_file, "w") as f:
            f.write("Test content")
            
        with open(test_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": (test_file, f, "text/plain")},
                headers=auth_headers
            )
            
        assert response.status_code == 200
        
        # Проверяем очистку всех временных файлов
        for temp_file in temp_files:
            assert not os.path.exists(temp_file)
            
    finally:
        # Гарантированная очистка
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@pytest.fixture(autouse=True)
async def cleanup_files():
    yield
    # Очистка после каждого теста
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
