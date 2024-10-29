import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.utils.logger import get_main_logger, get_rag_logger

# Initialize loggers for main and RAG-specific logging
logger = get_main_logger()
rag_logger = get_rag_logger()

def initialize_nltk():
    """Initialize NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

# Инициализируем NLTK при импорте модуля
initialize_nltk()

def process_large_file(file_path: str, chunk_size: int = 1000000) -> Generator[str, None, None]:
    """Process a large file in chunks of specified size."""
    logger.info(f"Processing large file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)  # Read a chunk of the file
            if not chunk:  # If no more content, exit the loop
                break
            yield chunk  # Yield the chunk for processing
    logger.info("Finished processing large file")  # Log completion of file processing

def extract_dates(text: Union[str, List[str]]) -> List[str]:
    """Extract dates from text or list of texts."""
    date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b'  # Regex pattern for date extraction
    
    if isinstance(text, list):
        # If input is a list, process each chunk and combine results
        all_dates = []
        for chunk in text:
            dates = re.findall(date_pattern, chunk)  # Find all dates in the chunk
            all_dates.extend(dates)  # Add found dates to the list
        return list(set(all_dates))  # Remove duplicates and return
    else:
        # If input is a single string
        return re.findall(date_pattern, text)  # Return found dates

def extract_named_entities(text: Union[str, List[str]]) -> List[str]:
    """Extract named entities from text or list of texts."""
    try:
        if isinstance(text, list):
            all_entities = []  # Initialize list for all entities
            for chunk in text:
                entities = _extract_entities_from_text(chunk) or []  # Handle None returns
                all_entities.extend(entities)  # Add found entities to the list
            return list(set(all_entities))  # Remove duplicates and return
        else:
            return _extract_entities_from_text(text) or []  # Handle None returns
    except Exception as e:
        logger.error(f"Error in extract_named_entities: {str(e)}")  # Log error
        return []  # Return empty list on error

def extract_key_phrases(text: Union[str, List[str]]) -> List[str]:
    """Extract key phrases from text or list of texts."""
    try:
        if isinstance(text, list):
            all_phrases = []  # Initialize list for all phrases
            for chunk in text:
                phrases = _extract_phrases_from_text(chunk) or []  # Handle None returns
                all_phrases.extend(phrases)  # Add found phrases to the list
            return list(set(all_phrases))  # Remove duplicates and return
        else:
            return _extract_phrases_from_text(text) or []  # Handle None returns
    except Exception as e:
        logger.error(f"Error in extract_key_phrases: {str(e)}")  # Log error
        return []  # Return empty list on error

def _extract_entities_from_text(text: str) -> List[str]:
    """Helper function to extract entities from a single text string."""
    try:
        # Your entity extraction logic here
        # For now, return empty list as placeholder
        return []  # Placeholder return
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")  # Log error
        return []  # Return empty list on error

def _extract_phrases_from_text(text: str) -> List[str]:
    """Helper function to extract phrases from a single text string."""
    try:
        # Your phrase extraction logic here
        # For now, return empty list as placeholder
        return []  # Placeholder return
    except Exception as e:
        logger.error(f"Error extracting phrases: {str(e)}")  # Log error
        return []  # Return empty list on error

def load_and_preprocess_text(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and preprocess text data."""
    try:
        # If input is already preprocessed
        if isinstance(input_data, dict):
            if 'text' not in input_data or 'chunks' not in input_data:
                raise ValueError("Preprocessed data must contain 'text' and 'chunks' keys")  # Validate keys
            return input_data  # Return preprocessed data
            
        # If input is raw text
        if not isinstance(input_data, str):
            raise ValueError(f"Expected string or preprocessed data dict, got {type(input_data)}")  # Validate input type
            
        # Process raw text
        text = input_data.strip()  # Strip whitespace from input
        if not text:
            raise ValueError("Empty text content")  # Validate non-empty content
            
        chunks = split_into_chunks(text)  # Split text into chunks
        
        processed_data = {
            'text': text,
            'chunks': chunks
        }
        
        logger.info(f"Text processed. Found {len(chunks)} chunks")  # Log number of chunks found
        rag_logger.info(
            f"\nText Processing:\n"
            f"Total chunks: {len(chunks)}\n"
            f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.2f} chars\n"
            f"Sample chunk: {chunks[0][:100]}...\n"
            f"{'-'*50}"
        )
        
        return processed_data  # Return processed data
        
    except Exception as e:
        error_msg = f"Error in load_and_preprocess_text: {str(e)}"  # Capture error message
        logger.error(error_msg)  # Log error
        rag_logger.error(f"\nProcessing Error:\n{error_msg}\n{'-'*50}")  # Log RAG error
        raise  # Raise the error

def split_into_chunks(text: Union[str, Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split the text into chunks with specified size and overlap."""
    logger.info(f"Input text type: {type(text)}")  # Log type of input text
    logger.info(f"Input text size: {len(text)} characters")  # Log size of input text
    
    if isinstance(text, dict):
        logger.info("Text is dictionary, extracting 'text' key")  # Log extraction from dictionary
        text = text.get('text', '')  # Extract text from dictionary
        if not text:
            logger.error("No text found in dictionary")  # Log error if no text found
            return []  # Return empty list
    
    words = text.split()  # Split text into words
    logger.info(f"Word count: {len(words)}")  # Log word count
    logger.info(f"Chunk size: {chunk_size}, overlap: {overlap}")  # Log chunk size and overlap
    
    chunks = []  # Initialize list for chunks
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])  # Create a chunk from words
        chunks.append(chunk)  # Add chunk to list
        
    logger.info(f"Created {len(chunks)} chunks")  # Log number of chunks created
    logger.info(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks) if chunks else 0} characters")  # Log average chunk size
    
    return chunks  # Return list of chunks

def read_file_content(file_path: str) -> str:
    """Read entire file content and return as string."""
    logger.info(f"Reading file: {file_path}")  # Log file reading
    full_text = []  # Initialize list for full text
    for chunk in process_large_file(file_path):
        full_text.append(chunk)  # Append each chunk to full text
    content = ''.join(full_text)  # Join all chunks into a single string
    logger.info(f"Read {len(content)} characters from file")  # Log number of characters read
    return content  # Return full text