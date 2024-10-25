from src.data_source import DataSource
from typing import List, Any

class ListDataSource(DataSource):
    def __init__(self, chunks: List[str], embeddings: List[List[float]], processed_text: Any = None):
        self._chunks = chunks
        self._embeddings = embeddings
        self._processed_text = processed_text

    def get_chunks(self) -> List[str]:
        return self._chunks

    def get_embeddings(self) -> List[List[float]]:
        return self._embeddings

    def get_processed_text(self) -> Any:
        return self._processed_text

