import os
import json
import firebase_admin
from firebase_admin import credentials, storage
import logging

logger = logging.getLogger(__name__)

class FirebaseStorageService:
    def __init__(self):
        try:
            # Получаем учетные данные из переменной окружения
            firebase_credentials_json = os.environ.get('FIREBASE_CREDENTIALS')
            
            if not firebase_credentials_json:
                raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")
            
            # Парсим JSON из строки
            service_account_info = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(service_account_info)
            
            # Инициализируем Firebase
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'your-project-id.appspot.com'
            })
            
            self.bucket = storage.bucket()
            logger.info("Firebase Storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Storage: {str(e)}")
            raise
