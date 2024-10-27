from src.web.app import create_app
from src.web.websocket import socketio
import os

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port)
