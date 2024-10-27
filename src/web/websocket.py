from flask_socketio import SocketIO
from typing import Dict, Any

socketio = SocketIO()

def emit_progress(status: str, current: int, total: int, additional_info: Dict[str, Any] = None):
    """Emit progress update through WebSocket."""
    data = {
        'status': status,
        'current': current,
        'total': total,
        'progress': (current / total * 100) if total > 0 else 0
    }
    if additional_info:
        data.update(additional_info)
    socketio.emit('progress_update', data)