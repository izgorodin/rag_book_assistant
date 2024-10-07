import re
from typing import List
from src.config import CHUNK_SIZE, OVERLAP
import logging

logger = logging.getLogger(__name__)

def extract_dates(text: str) -> List[str]:
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
    return re.findall(date_pattern, text)

def load_and_preprocess_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    dates = extract_dates(text)
    logger.info(f"Extracted {len(dates)} dates from the text")
    
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        if end > len(words):
            end = len(words)

        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

        if start >= len(words):
            break

    logger.info(f"Text split into {len(chunks)} chunks with chunk size {chunk_size} and overlap {overlap}.")
    return chunks