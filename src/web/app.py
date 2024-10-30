from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
import os
from typing import Optional

import uvicorn
from src.cli import BookAssistant
from src.utils.logger import get_main_logger, get_rag_logger
from src.file_processor import FileProcessor
from src.config import FLASK_SECRET_KEY
from src.web.websocket import WebSocketManager
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from .auth.middleware import AuthMiddleware
import aiofiles
import traceback
import uuid
from src.services.firebase_storage import FirebaseStorageService

# Initialize loggers
logger = get_main_logger()
rag_logger = get_rag_logger()

# Initialize FastAPI app
app = FastAPI(title="Book Assistant API")
app.auth_required = True  # Флаг для управления аутентификацией

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

# Инициализируем сервис
storage_service = FirebaseStorageService()

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

def get_storage_service():
    return storage_service

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user),
    storage_service: FirebaseStorageService = Depends(get_storage_service)
):
    temp_files = []
    try:
        logger.info("Upload started", extra={
            "user": user,
            "uploaded_file": file.filename,
            "content_type": file.content_type,
            "file_size": file.size
        })
        
        if not allowed_file(file.filename):
            logger.warning("Invalid file type rejected", extra={
                "user": user,
                "uploaded_file": file.filename,
                "allowed_extensions": list(ALLOWED_EXTENSIONS)
            })
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Сохраняем временный файл
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        temp_files.append(file_path)
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        logger.info("File saved", extra={
            "user": user,
            "file_location": file_path,
            "file_size": len(content)
        })
        
        # Загружаем файл в Firebase Storage
        try:
            storage_url = await storage_service.upload_file(file_path, user)
        except Exception as e:
            logger.error("Firebase upload error", extra={
                "user": user,
                "error": str(e)
            })
            raise HTTPException(
                status_code=500,
                detail=f"Firebase upload error: {str(e)}"
            )
        
        # Обработка книги
        try:
            book_data = assistant.load_and_process_book(file_path)
            app.state.book_data = book_data
            
            chunks_count = len(book_data.get_chunks())
            logger.info("Book processed", extra={
                "user": user,
                "chunks_count": chunks_count,
                "storage_url": storage_url
            })
            
            return JSONResponse({
                "status": "success",
                "message": "File processed successfully",
                "chunks_count": chunks_count,
                "storage_url": storage_url
            })
            
        except Exception as e:
            logger.error("Book processing error", extra={
                "user": user,
                "error": str(e)
            })
            app.state.book_data = None
            raise
            
    except Exception as e:
        # В случае ошибки тоже удаляем временный файл
        if temp_files and os.path.exists(temp_files[0]):
            os.remove(temp_files[0])
            logger.info("Temporary file removed after error", extra={
                "file_location": temp_files[0]
            })
        raise

@app.get("/ask")
async def ask_question(
    question: str,
    user: str = Depends(get_current_user)
):
    # Получаем book_data из состояния приложения
    book_data = getattr(app.state, 'book_data', None)
    
    if not book_data:
        logger.error("No book data loaded")
        raise HTTPException(
            status_code=400,
            detail="No book data loaded"
        )
    
    try:
        logger.info(f"Processing question: {question}")
        answer = assistant.answer_question(question, book_data)
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_book_loaded")
async def check_book_loaded(user: str = Depends(get_current_user)):
    book_data = getattr(app.state, 'book_data', None)
    return JSONResponse({'book_loaded': book_data is not None})

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await ws_manager.connect(websocket)
        
        while True:
            try:
                data = await websocket.receive_text()
                # Можно добавить обработку входящих сообщений если нужно
            except WebSocketDisconnect:
                await ws_manager.disconnect(websocket)
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await ws_manager.disconnect(websocket)
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    if username in USERS and USERS[username] == password:
        # Устанавливаем сессию
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"},
        status_code=401
    )

# Добавляем сессии (пере auth middleware!)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get('SESSION_SECRET_KEY', FLASK_SECRET_KEY),
    session_cookie="session",
    max_age=3600  # 1 hour
)

# Конфигурация CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get('ALLOWED_ORIGINS', 'http://localhost:8080').split(',')],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Добавляем middleware аутентификации с публичными путями
app.add_middleware(
    AuthMiddleware,
    public_paths=[
        "/login",
        "/static",
        "/docs",
        "/openapi.json",
        "/health",
        "/",  # Временно оставляем корневой путь публичным
        "/ws"  # WebSocket тоже публичный
    ]
)

@app.get("/")
async def home():
    return {"message": "Welcome to RAG Book Assistant"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Получаем порт из переменной окружения или используем значение по умолчанию
    PORT = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
