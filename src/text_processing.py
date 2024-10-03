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
    start = 0

    while start < len(words):
        end = start + chunk_size
        if end > len(words):
            end = len(words)  # Adjust end for the last chunk

        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Move the start index forward by chunk_size - overlap
        start += chunk_size - overlap

        # Ensure we don't skip past the end of the words list
        if start >= len(words):
            break

    return chunks