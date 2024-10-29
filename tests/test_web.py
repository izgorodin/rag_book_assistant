import pytest
from src.web.app import create_app
from unittest.mock import Mock

@pytest.fixture
def client():
    app = create_app(init_services=False)
    with app.test_client() as client:
        yield client

def test_login_success(client, caplog):
    """Test successful login with logging."""
    # Arrange
    credentials = {'username': 'admin', 'password': 'admin1q2w3e'}
    
    # Act
    response = client.post('/login', data=credentials)
    
    # Assert
    assert response.status_code == 302
    assert "User 'admin' logged in successfully" in caplog.text
