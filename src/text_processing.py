import re
from typing import List, Dict, Any, Union, Generator
from src.config import TEXT_PROCESSING_CONFIG
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.logger import setup_logger    
from src.types import (
    Chunk, ProcessedBook, TokensWithPOS, 
    TokenWithPOS, Context, ContextMetadata
)
from src.error_handler import handle_rag_error, DataSourceError, TokenizationError

logger = setup_logger()

@handle_rag_error
def process_large_file(file_path: str, chunk_size: int = 1000000) -> Generator[str, None, None]:
    """
    Process a large file in chunks of specified size.

    Args:
        file_path (str): Path to the file to be processed.
        chunk_size (int): Size of each chunk to read.

    Yields:
        str: Chunks of the file content.

    Raises:
        DataSourceError: If there's an issue reading the file.
    """
    logger.info(f"Processing large file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        logger.info("Finished processing large file")
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise DataSourceError(f"Error reading file: {str(e)}")

@handle_rag_error
def extract_dates(text: str) -> List[str]:
    """
    Extract dates from the given text using regex patterns.

    Args:
        text (str): Input text to extract dates from.

    Returns:
        List[str]: List of extracted dates.
    """
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
    return re.findall(date_pattern, text)

@handle_rag_error
def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from the text and categorize them.

    Args:
        text (str): Input text to extract named entities from.

    Returns:
        Dict[str, List[str]]: Dictionary of entity types and their occurrences.

    Raises:
        TokenizationError: If there's an issue with NLTK tokenization.
    """
    try:
        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        entities = {
            'PERSON': [], 'ORGANIZATION': [], 'LOCATION': [], 'DATE': [],
            'TIME': [], 'MONEY': [], 'PERCENT': [], 'FACILITY': [], 'GPE': []
        }
        
        for chunk in chunks:
            if isinstance(chunk, Tree):
                entity_type = chunk.label()
                entity_value = ' '.join([c[0] for c in chunk.leaves()])
                if entity_type in entities:
                    entities[entity_type].append(entity_value)
        
        return entities
    except nltk.exceptions.NLTKException as e:
        logger.error(f"NLTK error in named entity extraction: {str(e)}")
        raise TokenizationError(f"Error in named entity extraction: {str(e)}")

@handle_rag_error
def extract_key_phrases(text: str, num_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from the text based on part-of-speech tagging.

    Args:
        text (str): Input text to extract key phrases from.
        num_phrases (int): Number of key phrases to extract.

    Returns:
        List[str]: List of extracted key phrases.

    Raises:
        TokenizationError: If there's an issue with NLTK tokenization.
    """
    try:
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
    except nltk.exceptions.NLTKException as e:
        logger.error(f"NLTK error in key phrase extraction: {str(e)}")
        raise TokenizationError(f"Error in key phrase extraction: {str(e)}")

@handle_rag_error
def load_and_preprocess_text(file_path: str) -> ProcessedBook:
    """Load and preprocess text from file"""
    logger.info("Preprocessing text content")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Преобразуем в правильные типы
    chunks = [Chunk(chunk) for chunk in split_into_chunks(text_content)]
    dates = extract_dates(text_content)
    entities = extract_named_entities(text_content)
    key_phrases = extract_key_phrases(text_content)
    
    return {
        'chunks': chunks,
        'dates': dates,
        'entities': entities,
        'key_phrases': key_phrases
    }

@handle_rag_error
def split_into_chunks(text: str, chunk_size: int = TEXT_PROCESSING_CONFIG['chunk_size'], overlap: int = TEXT_PROCESSING_CONFIG['overlap']) -> List[Chunk]:
    """
    Split the text into chunks with specified size and overlap.

    Args:
        text (str): Input text to split into chunks.
        chunk_size (int): Size of each chunk.
        overlap (int): Overlap between chunks.

    Returns:
        List[Chunk]: List of text chunks.
    """
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
