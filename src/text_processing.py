import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.logger import setup_logger    
from src.types import Chunk

logger = setup_logger()


def process_large_file(file_path: str, chunk_size: int = 1000000) -> Generator[str, None, None]:
    """Process a large file in chunks of specified size."""
    logger.info(f"Processing large file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk
    logger.info("Finished processing large file")

def extract_dates(text: str) -> List[str]:
    """Extract dates from the given text using regex patterns."""
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
    return re.findall(date_pattern, text)

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from the text and categorize them."""
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = {
        'PERSON': [],
        'ORGANIZATION': [],
        'LOCATION': [],
        'DATE': [],
        'TIME': [],
        'MONEY': [],
        'PERCENT': [],
        'FACILITY': [],
        'GPE': []  # Geo-Political Entity
    }
    
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity_type = chunk.label()
            entity_value = ' '.join([c[0] for c in chunk.leaves()])
            if entity_type in entities:
                entities[entity_type].append(entity_value)
    
    return entities

def extract_key_phrases(text: str, num_phrases: int = 5) -> List[str]:
    """Extract key phrases from the text based on part-of-speech tagging."""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    grammar = r"""
        KP: {<JJ.*>*<NN.*>+}  # Key Phrases
        CP: {<JJ.*>*<NN.*>+<IN><JJ.*>*<NN.*>+}  # Complex Phrases
    """
    
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    
    phrases = []
    for subtree in tree.subtrees():
        if subtree.label() in ['KP', 'CP']:
            phrase = ' '.join([word for word, tag in subtree.leaves()])
            phrases.append(phrase)
    
    return sorted(set(phrases), key=phrases.count, reverse=True)[:num_phrases]

def load_and_preprocess_text(text_content: str) -> Dict[str, Any]:
    """Load and preprocess the text content, extracting relevant information."""
    logger.info("Preprocessing text content")
    
    dates = extract_dates(text_content)
    entities = extract_named_entities(text_content)
    key_phrases = extract_key_phrases(text_content)
    
    logger.info(f"Extracted {len(dates)} dates, {sum(len(v) for v in entities.values())} named entities, and {len(key_phrases)} key phrases from the text")
    
    chunks = split_into_chunks(text_content)
    
    return {
        'text': text_content,
        'chunks': chunks,
        'dates': dates,
        'entities': entities,
        'key_phrases': key_phrases
    }

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Chunk]:
    """Split the text into chunks with specified size and overlap."""
    logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {overlap})")
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunk = text[start:end]
        chunks.append(Chunk(chunk))
        start = end - overlap
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

BOOK_PATH = 'tests/data/book.txt'
FORD_PATH = 'tests/data/ford.txt'
