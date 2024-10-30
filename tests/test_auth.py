import pytest
from fastapi.testclient import TestClient
from src.web.app import app

def test_public_routes(client):
    """Test access to public routes."""
    routes = [
        "/login",
        "/health",
        "/docs",
        "/openapi.json"
    ]
    
    for route in routes:
        response = client.get(route)
        assert response.status_code != 401, f"Route {route} should be public"

def test_protected_routes_without_auth(client):
    """Test access to protected routes without authentication."""
    routes = [
        "/upload",
        "/ask"
    ]
    
    for route in routes:
        response = client.get(route)
        assert response.status_code == 401, f"Route {route} should require authentication" 