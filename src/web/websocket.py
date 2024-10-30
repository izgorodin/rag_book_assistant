from fastapi import WebSocket
import logging
from typing import List, Dict, Any

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def emit_progress(self, status: str, current: int, total: int, additional_data: Dict[str, Any] = None):
        if not self.active_connections:
            return
            
        message = {
            "type": "progress",
            "status": status,
            "current": current,
            "total": total
        }
        if additional_data:
            message.update(additional_data)
        
        for connection in self.active_connections.copy():  # Используем copy() для безопасной итерации
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending WebSocket message: {str(e)}")
                await self.disconnect(connection)