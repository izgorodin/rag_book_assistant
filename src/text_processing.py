import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.logger import setup_logger    

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

def load_and_preprocess_text(file_path: str) -> Dict[str, Any]:
    """Load and preprocess the text content from file."""
    logger.info(f"Loading and preprocessing file: {file_path}")
    
    # Read full content
    text_content = read_file_content(file_path)
    if not text_content:
        logger.error("Empty file content")
        raise ValueError("Empty file content")
        
    # Process text
    dates = extract_dates(text_content)
    entities = extract_named_entities(text_content)
    key_phrases = extract_key_phrases(text_content)
    chunks = split_into_chunks(text_content)
    
    return {
        'text': text_content,
        'chunks': chunks,
        'dates': dates,
        'entities': entities,
        'key_phrases': key_phrases
    }

def split_into_chunks(text: Union[str, Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split the text into chunks with specified size and overlap."""
    logger.info(f"Input text type: {type(text)}")
    logger.info(f"Input text size: {len(text)} characters")
    
    if isinstance(text, dict):
        logger.info("Text is dictionary, extracting 'text' key")
        text = text.get('text', '')
        if not text:
            logger.error("No text found in dictionary")
            return []
    
    words = text.split()
    logger.info(f"Word count: {len(words)}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {overlap}")
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    logger.info(f"Created {len(chunks)} chunks")
    logger.info(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks) if chunks else 0} characters")
    
    return chunks


def read_file_content(file_path: str) -> str:
    """Read entire file content and return as string."""
    logger.info(f"Reading file: {file_path}")
    full_text = []
    for chunk in process_large_file(file_path):
        full_text.append(chunk)
    content = ''.join(full_text)
    logger.info(f"Read {len(content)} characters from file")
    return content
