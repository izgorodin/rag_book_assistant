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
from src.cli import BookAssistant
from src.utils.logger import get_main_logger, get_rag_logger
from src.file_processor import FileProcessor
from src.config import FLASK_SECRET_KEY
from src.web.websocket import WebSocketManager
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer

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
    
    try:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Начало загрузки
        await ws_manager.emit_progress("Starting upload", 0, 100)
        
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Файл загружен
        await ws_manager.emit_progress("File uploaded", 25, 100)
        
        # Обработка книги
        await ws_manager.emit_progress("Processing book", 50, 100)
        book_data = assistant.load_and_process_book(file_path)
        
        # Завершение
        chunks_count = len(book_data.get_chunks())
        await ws_manager.emit_progress("Complete", 100, 100, {
            "chunks_count": chunks_count
        })
        
        return JSONResponse({
            "status": "success",
            "message": "File processed successfully",
            "chunks_count": chunks_count
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        await ws_manager.emit_progress("Error", 0, 100, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(question: str = Query(...), user: str = Depends(get_current_user)):
    try:
        if not book_data:
            raise HTTPException(status_code=400, detail="No book data loaded")
            
        # Отправляем статус через WebSocket
        await ws_manager.emit_progress("Processing question", 0, 100)
        
        # Получаем ответ (исправлено имя метода)
        answer = assistant.answer_question(question, book_data)
        
        # Отправляем завершение через WebSocket
        await ws_manager.emit_progress("Complete", 100, 100, {
            "answer": answer
        })
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await ws_manager.emit_progress("Error", 0, 100, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_book_loaded")
async def check_book_loaded(user: str = Depends(get_current_user)):
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
        response = RedirectResponse(url="/", status_code=302)
        # Здесь можно добавить установку cookie или сессии
        return response
    
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"},
        status_code=401
    )

# Middleware для проверки аутентификации
@app.middleware("http")
async def auth_middleware(request, call_next):
    if app.auth_required:
        # Проверка аутентификации
        if not is_authenticated(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
    response = await call_next(request)
    return response

def is_authenticated(request):
    # Логика проверки аутентификации
    return True if not app.auth_required else False  # Для тестов всегда True если auth_required=False
