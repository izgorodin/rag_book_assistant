from typing import List, Dict, Any
from src.types import Chunk, EmbeddingList
import pickle
import os
from src.data_source import DataSource
from src.error_handler import handle_rag_error, DataSourceError

class BookDataInterface(DataSource):
    def __init__(self, chunks: List[Chunk], embeddings: EmbeddingList, processed_text: Dict[str, Any]):
        self._chunks: List[Chunk] = chunks
        self._embeddings: EmbeddingList = embeddings
        self._processed_text: Dict[str, Any] = processed_text

    @classmethod
    @handle_rag_error
    def from_file(cls, file_path: str) -> 'BookDataInterface':
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return cls(data['chunks'], data['embeddings'], data.get('processed_text', {}))
            elif isinstance(data, BookDataInterface):
                return data
            else:
                raise ValueError(f"Unexpected data type in file: {type(data)}")
        except Exception as e:
            raise DataSourceError(f"Error loading from file {file_path}: {str(e)}")

    @handle_rag_error
    def save(self, file_path: str):
        try:
            data = {
                'chunks': self._chunks,
                'embeddings': self._embeddings,
                'processed_text': self._processed_text
            }
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise DataSourceError(f"Error saving to file {file_path}: {str(e)}")

    def __len__(self) -> int:
        return len(self._chunks)

    def get_chunks(self) -> List[Chunk]:
        return self._chunks

    def get_embeddings(self) -> EmbeddingList:
        return self._embeddings

    def get_processed_text(self) -> Dict[str, Any]:
        return self._processed_text
