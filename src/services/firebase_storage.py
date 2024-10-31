import firebase_admin
from firebase_admin import credentials, storage
import os
from src.utils.logger import get_main_logger



logger = get_main_logger()

class FirebaseStorageService:
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
            })
        self.bucket = storage.bucket()
        
    async def upload_file(self, file_path: str, user: str) -> str:
        """Загружает файл в Firebase Storage и возвращает URL"""
        try:
            file_name = os.path.basename(file_path)
            blob_path = f"uploads/{user}/{file_name}"
            blob = self.bucket.blob(blob_path)
            
            logger.info("Starting Firebase upload", extra={
                "user": user,
                "file_name": file_name
            })
            
            blob.upload_from_filename(file_path)
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
                "error": str(e)
            })
            raise
