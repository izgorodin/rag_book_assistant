import re
from typing import List
from src.config import CHUNK_SIZE, OVERLAP

def load_and_preprocess_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters, keep only letters, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > chunk_size:
            chunk_words = chunk.split()
            while len(' '.join(chunk_words)) > chunk_size:
                chunk_words.pop()
            chunk = ' '.join(chunk_words)
        chunks.append(chunk)
    return chunks