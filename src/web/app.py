from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
import os
from typing import Optional
from src.cli import BookAssistant
from src.utils.logger import get_main_logger, get_rag_logger
from src.file_processor import FileProcessor
from src.config import FLASK_SECRET_KEY
from src.web.websocket import WebSocketManager

# Initialize loggers
logger = get_main_logger()
rag_logger = get_rag_logger()

# Initialize FastAPI app
app = FastAPI(title="Book Assistant API")

# Configure static files and templates
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt'}

# Configure authentication
security = HTTPBasic()
USERS = {
    'admin': os.environ.get('ADMIN_PASSWORD', 'admin1q2w3e'),
    'tester1': os.environ.get('TESTER_PASSWORD', '41dsf3qw7sDa')
}

# Initialize services
assistant = BookAssistant(progress_callback=ws_manager.emit_progress)
file_processor = FileProcessor()
book_data = None

async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username in USERS and USERS[credentials.username] == credentials.password:
        return credentials.username
    raise HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Добавим функцию для работы с url в шаблонах
def url_for(request: Request, name: str, **path_params):
    return request.url_for(name, **path_params)

@app.get("/")
async def index(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "url_for": lambda name, **path_params: request.url_for(name, **path_params)
        }
    )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):
    global book_data
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process book
        await ws_manager.emit_progress("Starting file processing", 0, 100)
        book_data = assistant.load_and_process_book(file_path)
        
        chunks_count = len(book_data.get_chunks())
        await ws_manager.emit_progress("Processing complete", 100, 100, {
            'chunks_count': chunks_count
        })
        
        return JSONResponse({
            'status': 'success',
            'message': 'File processed successfully',
            'chunks_count': chunks_count
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        await ws_manager.emit_progress("Error", 0, 100, {'error': str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    user: str = Depends(get_current_user)
):
    global book_data
    
    if not book_data:
        raise HTTPException(status_code=400, detail="No book loaded")
    
    try:
        logger.info(f"Processing question: {question}")
        answer = assistant.answer_question(question, book_data)
        logger.info("Answer generated successfully")
        return JSONResponse({
            'status': 'success',
            'answer': answer
        })
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_book_loaded")
async def check_book_loaded(user: str = Depends(get_current_user)):
    return JSONResponse({'book_loaded': book_data is not None})

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle WebSocket messages if needed
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await ws_manager.disconnect(websocket)
