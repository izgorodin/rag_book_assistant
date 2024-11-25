import epub2txt
import logging
from typing import Optional, List
import asyncio
from src.services.batch_processor import BatchProcessor
from src.config import BATCH_SIZES, BATCH_SETTINGS

logger = logging.getLogger(__name__)

class EPUBProcessor:
    """Процессор для извлечения текста из EPUB файлов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.batch_processor = BatchProcessor(
            batch_size=BATCH_SIZES['text_chunks'],
            max_workers=BATCH_SETTINGS['max_workers']
        )

    async def process_epub(self, file_path: str, progress_callback=None) -> str:
        """Обработка EPUB файла с батчингом для больших файлов."""
        try:
            text = await asyncio.to_thread(epub2txt.epub2txt, file_path)
            
            if len(text) > BATCH_SIZES['text_chunks']:
                chunks = self.batch_processor.chunk_text(
                    text=text,
                    chunk_size=BATCH_SIZES['text_chunks'],
                    overlap=BATCH_SETTINGS['chunk_overlap']
                )
                
                async def process_chunks(chunks: List[str]) -> List[str]:
                    return [await asyncio.to_thread(self._process_chunk, chunk) for chunk in chunks]
                
                processed_chunks = await self.batch_processor.process_async(
                    items=chunks,
                    processor=process_chunks,
                    description="Processing EPUB chunks",
                    progress_callback=progress_callback
                )
                
                return ''.join(processed_chunks)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to process EPUB file: {str(e)}")
            raise ValueError(f"Error processing EPUB file: {str(e)}")

    async def _process_chunk(self, chunk: str) -> str:
        """Обработка отдельного чанка текста."""
        # Здесь может быть дополнительная обработка чанка
        return chunk
