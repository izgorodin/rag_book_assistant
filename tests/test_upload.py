import pytest
from fastapi.testclient import TestClient
from src.web.app import app
import os
import base64
from src.config import UPLOAD_FOLDER
from src.book_data_interface import BookDataInterface

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

def test_auth_required(client):
    """Проверка что без авторизации доступ запрещен"""
    response = client.post("/upload")
    assert response.status_code == 401

def test_file_upload_success(client, auth_headers, test_file):
    """Проверка успешной загрузки файла"""
    with open(test_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": (test_file, f, "text/plain")},
            headers=auth_headers
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Проверяем что файл сохранился
    uploaded_path = os.path.join(UPLOAD_FOLDER, test_file)
    assert os.path.exists(uploaded_path)

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

def test_cleanup(client, auth_headers, test_file):
    """Проверка очистки временных файлов"""
    # Загружаем файл
    with open(test_file, "rb") as f:
        client.post(
            "/upload",
            files={"file": (test_file, f, "text/plain")},
            headers=auth_headers
        )
    
    uploaded_path = os.path.join(UPLOAD_FOLDER, test_file)
    assert os.path.exists(uploaded_path)
    
    # После теста проверяем очистку
    os.remove(uploaded_path)
    assert not os.path.exists(uploaded_path)
