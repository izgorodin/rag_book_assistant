import pytest
from src.web.websocket import WebSocketManager
from fastapi import WebSocket
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def ws_manager():
    return WebSocketManager()

@pytest.fixture
def mock_websocket():
    ws = Mock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws

@pytest.mark.asyncio
async def test_connect(ws_manager, mock_websocket):
    """Test WebSocket connection"""
    client_id = "test_client"
    await ws_manager.connect(mock_websocket, client_id)
    
    mock_websocket.accept.assert_called_once()
    assert mock_websocket in ws_manager.active_connections[client_id]

@pytest.mark.asyncio
async def test_disconnect(ws_manager, mock_websocket):
    """Test WebSocket disconnection"""
    client_id = "test_client"
    await ws_manager.connect(mock_websocket, client_id)
    await ws_manager.disconnect(mock_websocket, client_id)
    
    assert client_id not in ws_manager.active_connections

@pytest.mark.asyncio
async def test_emit_progress(ws_manager, mock_websocket):
    """Test progress emission"""
    client_id = "test_client"
    await ws_manager.connect(mock_websocket, client_id)
    
    progress_data = {
        "status": "Processing",
        "current": 50,
        "total": 100
    }
    
    await ws_manager.emit_progress(**progress_data, client_id=client_id)
    
    mock_websocket.send_json.assert_called_once()
    called_data = mock_websocket.send_json.call_args[0][0]
    assert called_data["type"] == "progress"
    assert called_data["progress"] == 50.0 