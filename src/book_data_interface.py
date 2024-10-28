from typing import List, Dict, Any, Optional
import pickle
import os
from src.data_source import DataSource
from src.embedding import EmbeddingService

class BookDataInterface(DataSource):
    def __init__(self, 
                 chunks: List[str], 
                 embeddings: List[List[float]], 
                 processed_text: Dict[str, Any],
                 embedding_service: EmbeddingService,
                 dates: Optional[List[str]] = None,
                 entities: Optional[List[Dict[str, Any]]] = None,
                 key_phrases: Optional[List[str]] = None):
        self._chunks = chunks
        self._embeddings = embeddings
        self._processed_text = processed_text
        self._embedding_service = embedding_service
        self._dates = dates or []
        self._entities = entities or []
        self._key_phrases = key_phrases or []
        
    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['chunks'], data['embeddings'], data.get('processed_text', {}), data.get('embedding_service', {}), data.get('dates', []), data.get('entities', []), data.get('key_phrases', []))

    def save(self, file_path: str):
        data = {
            'chunks': self._chunks,
            'embeddings': self._embeddings,
            'processed_text': self._processed_text,
            'dates': self._dates,
            'entities': self._entities,
            'key_phrases': self._key_phrases
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

    def get_dates(self) -> List[str]:
        return self._dates

    def get_entities(self) -> List[Dict[str, Any]]:
        return self._entities

    def get_key_phrases(self) -> List[str]:
        return self._key_phrases

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_service.create_embeddings(texts)
