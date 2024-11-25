import pytest
from src.services.batch_processor import BatchProcessor

@pytest.fixture
def batch_processor():
    return BatchProcessor(batch_size=2, max_workers=2)

@pytest.mark.asyncio
async def test_process_async():
    processor = BatchProcessor(batch_size=2)
    
    async def mock_processor(items):
        return [item * 2 for item in items]
    
    items = [1, 2, 3, 4, 5]
    result = await processor.process_async(
        items=items,
        processor=mock_processor,
        description="Test processing"
    )
    
    assert result == [2, 4, 6, 8, 10]

def test_process_sync():
    processor = BatchProcessor(batch_size=2)
    
    def mock_processor(item):
        return item * 2
    
    items = [1, 2, 3, 4, 5]
    result = processor.process_sync(
        items=items,
        processor=mock_processor,
        description="Test processing"
    )
    
    assert result == [2, 4, 6, 8, 10]

def test_chunk_text():
    processor = BatchProcessor(batch_size=2)
    text = "This is a test text for chunking"
    chunks = processor.chunk_text(text=text, chunk_size=2, overlap=1)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "This is" in chunks[0]

@pytest.mark.asyncio
async def test_process_async_with_progress():
    processor = BatchProcessor(batch_size=2)
    progress_calls = []
    
    def progress_callback(info):
        progress_calls.append(info)
    
    async def mock_processor(items):
        return [item * 2 for item in items]
    
    items = [1, 2, 3, 4]
    await processor.process_async(
        items=items,
        processor=mock_processor,
        description="Test processing",
        progress_callback=progress_callback
    )
    
    assert len(progress_calls) > 0
    assert all('progress' in call for call in progress_calls) 