import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import logging
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

logger = logging.getLogger(__name__)

def process_large_file(file_path: str, chunk_size: int = 1000000) -> Generator[str, None, None]:
    logger.info(f"Processing large file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk
    logger.info("Finished processing large file")

def extract_dates(text: str) -> List[str]:
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
    return re.findall(date_pattern, text)

def extract_named_entities(text: str) -> Dict[str, List[str]]:
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
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    grammar = r"""
        KP: {<JJ.*>*<NN.*>+}
        CP: {<JJ.*>*<NN.*>+<IN><JJ.*>*<NN.*>+}
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
    logger.info("Preprocessing text content")
    chunks = split_into_chunks(text_content)
    
    dates = extract_dates(text_content)
    entities = extract_named_entities(text_content)
    key_phrases = extract_key_phrases(text_content)
    
    logger.info(f"Extracted {len(dates)} dates, {sum(len(v) for v in entities.values())} named entities, and {len(key_phrases)} key phrases from the text")
    
    return {
        'chunks': chunks,
        'dates': dates,
        'entities': entities,
        'key_phrases': key_phrases
    }

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {overlap})")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks
