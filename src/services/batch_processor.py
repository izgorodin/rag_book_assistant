from typing import TypeVar, Generic, List, Callable, AsyncIterator, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.utils.logger import get_main_logger

T = TypeVar('T')
R = TypeVar('R')

logger = get_main_logger()

class BatchProcessor(Generic[T, R]):
    """Universal batch processor for both async and sync operations."""
    
    def __init__(self, batch_size: int, max_workers: int = 8):
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    async def process_async(
        self,
        items: List[T],
        processor: Callable[[List[T]], List[R]] | Callable[[List[T]], AsyncIterator[R]],
        description: str = "",
        progress_callback: Callable = None
    ) -> List[R]:
        """Process items in batches asynchronously."""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]
            
            try:
                batch_result = processor(batch)
                if asyncio.iscoroutine(batch_result):
                    batch_result = await batch_result
                elif isinstance(batch_result, AsyncIterator):
                    batch_result = [item async for item in batch_result]
                
                results.extend(batch_result)
                
                if progress_callback:
                    progress_callback({
                        'stage': description,
                        'progress': (batch_idx + 1) / total_batches * 100,
                        'current': batch_idx + 1,
                        'total': total_batches
                    })
                
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                raise
                
        return results

    def process_sync(
        self,
        items: List[T],
        processor: Callable[[T], R],
        description: str = "",
        progress_callback: Callable = None
    ) -> List[R]:
        """Process items in parallel using ThreadPoolExecutor."""
        total_items = len(items)
        
        def process_with_progress(item_data: tuple[int, T]) -> R:
            idx, item = item_data
            result = processor(item)
            
            if progress_callback:
                progress_callback({
                    'stage': description,
                    'progress': (idx + 1) / total_items * 100,
                    'current': idx + 1,
                    'total': total_items
                })
                
            return result

        with ThreadPoolExecutor(max_workers=min(self.max_workers, total_items)) as executor:
            results = list(tqdm(
                executor.map(process_with_progress, enumerate(items)),
                total=total_items,
                desc=description,
                unit="item"
            ))
            
        return results

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
            
        return chunks 