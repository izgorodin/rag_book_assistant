from typing import List, Dict, Any
import pickle
import os
from src.data_source import DataSource

class BookDataInterface(DataSource):
    def __init__(self, chunks: List[str], embeddings: List[List[float]], 
                 processed_text: Dict[str, Any], embedding_service: Any):
        self._chunks = chunks
        self._embeddings = embeddings
        self._processed_text = processed_text
        self._embedding_service = embedding_service
        
    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['chunks'], data['embeddings'], data.get('processed_text', {}))

    def save(self, file_path: str):
        data = {
            'chunks': self._chunks,
            'embeddings': self._embeddings,
            'processed_text': self._processed_text
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self):
        return len(self._chunks)

    def get_chunks(self) -> List[str]:
        return self._chunks

    def get_embeddings(self) -> List[List[float]]:
        return self._embeddings

    def get_processed_text(self) -> Dict[str, Any]:
        return self._processed_text

    def create_embedding(self, text: str) -> List[float]:
        return self._embedding_service.create_embedding(text)
