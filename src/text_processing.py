import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.utils.logger import setup_logger    

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

def extract_dates(text: Union[str, List[str]]) -> List[str]:
    """Extract dates from text or list of texts."""
    date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b'
    
    if isinstance(text, list):
        # If input is a list, process each chunk and combine results
        all_dates = []
        for chunk in text:
            dates = re.findall(date_pattern, chunk)
            all_dates.extend(dates)
        return list(set(all_dates))  # Remove duplicates
    else:
        # If input is a single string
        return re.findall(date_pattern, text)

def extract_named_entities(text: Union[str, List[str]]) -> List[str]:
    """Extract named entities from text or list of texts."""
    try:
        if isinstance(text, list):
            all_entities = []
            for chunk in text:
                entities = _extract_entities_from_text(chunk) or []  # Handle None returns
                all_entities.extend(entities)
            return list(set(all_entities))
        else:
            return _extract_entities_from_text(text) or []  # Handle None returns
    except Exception as e:
        logger.error(f"Error in extract_named_entities: {str(e)}")
        return []  # Return empty list on error

def extract_key_phrases(text: Union[str, List[str]]) -> List[str]:
    """Extract key phrases from text or list of texts."""
    try:
        if isinstance(text, list):
            all_phrases = []
            for chunk in text:
                phrases = _extract_phrases_from_text(chunk) or []  # Handle None returns
                all_phrases.extend(phrases)
            return list(set(all_phrases))
        else:
            return _extract_phrases_from_text(text) or []  # Handle None returns
    except Exception as e:
        logger.error(f"Error in extract_key_phrases: {str(e)}")
        return []  # Return empty list on error

def _extract_entities_from_text(text: str) -> List[str]:
    """Helper function to extract entities from a single text string."""
    try:
        # Your entity extraction logic here
        # For now, return empty list as placeholder
        return []
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return []  # Return empty list on error

def _extract_phrases_from_text(text: str) -> List[str]:
    """Helper function to extract phrases from a single text string."""
    try:
        # Your phrase extraction logic here
        # For now, return empty list as placeholder
        return []
    except Exception as e:
        logger.error(f"Error extracting phrases: {str(e)}")
        return []  # Return empty list on error

def load_and_preprocess_text(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and preprocess text data."""
    try:
        # If input is already preprocessed
        if isinstance(input_data, dict):
            if 'text' not in input_data or 'chunks' not in input_data:
                raise ValueError("Preprocessed data must contain 'text' and 'chunks' keys")
            return input_data
            
        # If input is raw text
        if not isinstance(input_data, str):
            raise ValueError(f"Expected string or preprocessed data dict, got {type(input_data)}")
            
        # Process raw text
        text = input_data.strip()
        if not text:
            raise ValueError("Empty text content")
            
        chunks = split_into_chunks(text)
        
        processed_data = {
            'text': text,
            'chunks': chunks
        }
        
        logger.info(f"Text processed. Found {len(chunks)} chunks")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_text: {str(e)}")
        raise

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
