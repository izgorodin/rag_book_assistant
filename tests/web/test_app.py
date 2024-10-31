import pytest
from fastapi.testclient import TestClient
from src.web.app import app
import os
import base64

client = TestClient(app)

def get_auth_header(username: str, password: str) -> dict:
    """Helper function to create auth header"""
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}

@pytest.fixture
def auth_headers():
    """Fixture for authentication headers"""
    return get_auth_header("admin", "admin1q2w3e")

@pytest.fixture
def test_file(tmp_path):
    """Fixture to create a test file"""
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("Test content for the book")
    return file_path

def test_index_page():
    """Test the index page loads"""
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Book QA System" in response.text

def test_upload_no_auth():
    """Test upload endpoint requires authentication"""
    response = client.post("/upload")
    assert response.status_code == 401

def test_upload_with_auth_no_file(auth_headers):
    """Test upload endpoint requires file"""
    response = client.post("/upload", headers=auth_headers)
    assert response.status_code == 422

def test_upload_invalid_file_type(auth_headers):
    """Test upload rejects invalid file types"""
    files = {"file": ("test.xyz", "content", "text/plain")}
    response = client.post("/upload", headers=auth_headers, files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_upload_valid_file(auth_headers, test_file):
    """Test successful file upload"""
    with open(test_file, "rb") as f:
        files = {"file": ("test.txt", f, "text/plain")}
        response = client.post("/upload", headers=auth_headers, files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_ask_question_no_book(auth_headers):
    """Test ask endpoint requires book upload first"""
    response = client.post("/ask", headers=auth_headers, json={"question": "test question"})
    assert response.status_code == 400
    assert "No book data available" in response.json()["detail"]

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection"""
    with client.websocket_connect("/ws") as websocket:
        assert websocket.accepted
        # Test disconnection
        websocket.close()

def test_allowed_file():
    """Test file extension validation"""
    from src.web.app import allowed_file
    
    assert allowed_file("test.txt") == True
    assert allowed_file("test.pdf") == True
    assert allowed_file("test.xyz") == False
    assert allowed_file("test") == False

def test_get_current_user():
    """Test user authentication"""
    from src.web.app import get_current_user
    from fastapi import HTTPBasicCredentials
    
    valid_creds = HTTPBasicCredentials(username="admin", password="admin1q2w3e")
    invalid_creds = HTTPBasicCredentials(username="wrong", password="wrong")
    
    assert get_current_user(valid_creds) == "admin"
    
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(invalid_creds)
    assert exc_info.value.status_code == 401
