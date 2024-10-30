import pytest
from fastapi.testclient import TestClient
from src.web.app import app

# Временно отключаем аутентификацию для тестов
app.auth_required = False  # Нужно добавить эту опцию в app.py

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_home_page(client):
    """Test home page access."""
    response = client.get('/')
    assert response.status_code == 200

def test_upload_file(client):
    """Test file upload."""
    file_content = b"Test book content"
    files = {
        'file': ('test.txt', file_content, 'text/plain')
    }
    
    response = client.post('/upload', files=files)
    assert response.status_code == 200
    assert response.json()['status'] == 'success'

def test_ask_question_no_book(client):
    """Test asking question without uploading book."""
    response = client.get('/ask?question=test')
    assert response.status_code == 400
    assert 'No book data loaded' in response.json()['detail']

@pytest.mark.asyncio
async def test_websocket(client):
    """Test WebSocket connection."""
    with client.websocket_connect('/ws') as websocket:
        # Отправляем тестовое сообщение
        websocket.send_text("test")
        # Проверяем что соединение работает
        data = websocket.receive_text()
        assert data is not None
