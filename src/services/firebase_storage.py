import os
import json
import firebase_admin
from firebase_admin import credentials, storage
import logging
import uuid

logger = logging.getLogger(__name__)

class FirebaseStorageService:
    def __init__(self):
        try:
            # Получаем путь к файлу credentials
            credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
            if not credentials_path or not os.path.exists(credentials_path):
                raise ValueError(f"Firebase credentials file not found at {credentials_path}")
            
            # Инициализируем Firebase с credentials из файла
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
            })
            
            self.bucket = storage.bucket()
            logger.info("Firebase Storage initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Firebase Storage", extra={
                "error": str(e),
                "credentials_path": credentials_path
            })
            raise

    async def upload_file(self, file_path: str, user: str) -> str:
        """
        Загружает файл в Firebase Storage и возвращает URL.
        
        Args:
            file_path: Путь к локальному файлу
            user: Идентификатор пользователя
        
        Returns:
            str: Публичный URL загруженного файла
        
        Raises:
            Exception: При ошибке загрузки
        """
        try:
            file_name = os.path.basename(file_path)
            # Создаем уникальный путь для файла
            blob_path = f"uploads/{user}/{uuid.uuid4()}_{file_name}"
            blob = self.bucket.blob(blob_path)
            
            logger.info("Starting Firebase upload", extra={
                "user": user,
                "file_name": file_name,
                "blob_path": blob_path
            })
            
            # Загружаем файл
            blob.upload_from_filename(file_path)
            
            # Делаем файл публично доступным
            blob.make_public()
            
            url = blob.public_url
            logger.info("File uploaded to Firebase", extra={
                "user": user,
                "url": url
            })
            
            return url
                
        except Exception as e:
            logger.error("Firebase upload error", extra={
                "user": user,
                "file_path": file_path,
                "error": str(e)
            })
            raise
