import re
from typing import List, Dict, Any
from src.config import CHUNK_SIZE, OVERLAP
import logging
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

logger = logging.getLogger(__name__)

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

def load_and_preprocess_text(file_path: str) -> Dict[str, any]:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    dates = extract_dates(text)
    entities = extract_named_entities(text)
    key_phrases = extract_key_phrases(text)
    
    logger.info(f"Extracted {len(dates)} dates, {sum(len(v) for v in entities.values())} named entities, and {len(key_phrases)} key phrases from the text")
    
    return {
        'text': text.strip(),
        'dates': dates,
        'entities': entities,
        'key_phrases': key_phrases
    }

def split_into_chunks(text: Dict[str, Any], chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    words = text['text'].split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
