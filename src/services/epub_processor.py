import epub2txt
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EPUBProcessor:
    """Процессор для извлечения текста из EPUB файлов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_epub(self, file_path: str) -> str:
        """
        Извлекает текст из EPUB файла
        
        Args:
            file_path (str): Путь к EPUB файлу
            
        Returns:
            str: Извлеченный текст
            
        Raises:
            ValueError: Если произошла ошибка при обработке файла
        """
        try:
            self.logger.info(f"Processing EPUB file: {file_path}")
            text = epub2txt.epub2txt(file_path)
            self.logger.info(f"Successfully extracted {len(text)} characters")
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to process EPUB file: {str(e)}")
            raise ValueError(f"Error processing EPUB file: {str(e)}")
