from fastapi import WebSocket
import logging
from typing import List, Dict, Any

class WebSocketManager:
    def __init__(self, timeout: int = 300):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"New WebSocket connection. Active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def emit_progress(self, status: str = "", current: int = 0, total: int = 100):
        """
        Отправляет обновление прогресса через WebSocket
        """
        data = {
            "type": "progress",
            "status": status,
            "progress": (current / total) * 100 if total > 0 else 0,
            "current": current,
            "total": total
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                self.logger.error(f"Error sending WebSocket message: {str(e)}")
                await self.disconnect(connection)