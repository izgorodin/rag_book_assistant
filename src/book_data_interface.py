from typing import List, Dict, Any
import pickle
import os

class BookDataInterface:
    def __init__(self, chunks: List[str], embeddings: List[List[float]], processed_text: Dict[str, Any]):
        self.chunks = chunks
        self.embeddings = embeddings
        self.processed_text = processed_text

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['chunks'], data['embeddings'], data['processed_text'])

    def save(self, file_path: str):
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'processed_text': self.processed_text
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self):
        return len(self.chunks)
