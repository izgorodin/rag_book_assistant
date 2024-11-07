from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from src.utils.logger import get_rag_logger
from src.cli import BookAssistant
import os
import aiofiles

# Инициализация FastAPI приложения
app = FastAPI(
    title="Book Assistant API",
    description="REST API for RAG-powered Book Assistant",
    version="1.0.0"
)

# Инициализация логгера и ассистента
rag_logger = get_rag_logger()
assistant = BookAssistant()

# Настройки для загрузки файлов
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt', 'epub'}

# Модели данных
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Вспомогательные функции
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API endpoints
@app.post("/api/v1/upload")
async def upload_file(file: UploadFile):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    async with aiofiles.open(file_path, 'wb') as f:
        while content := await file.read(1024):
            await f.write(content)
    return JSONResponse(content={"message": "File uploaded successfully"})
