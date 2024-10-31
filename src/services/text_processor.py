import re
from typing import List, Dict, Any, Union, Generator
from src.config import CHUNK_SIZE, OVERLAP
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from src.utils.logger import get_main_logger, get_rag_logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
import asyncio

# Initialize loggers for main and RAG-specific logging
logger = get_main_logger()
rag_logger = get_rag_logger()

def initialize_nltk():
    """Initialize NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('averaged_perceptron_tagger')
        nltk.data.find('maxent_ne_chunker')
        nltk.data.find('words')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)

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
        # Токенизация и POS-тегирование
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Извлечение именованных сущностей
        chunks = ne_chunk(tagged)
        
        entities = []
        for chunk in chunks:
            if isinstance(chunk, Tree):
                # Извлекаем текст сущности и её тип
                entity_text = ' '.join([token for token, pos in chunk.leaves()])
                entity_type = chunk.label()
                entities.append(f"{entity_text} ({entity_type})")
        
        return list(set(entities))  # Убираем дубликаты
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return []

def _extract_phrases_from_text(text: str) -> List[str]:
    """Helper function to extract phrases from a single text string."""
    try:
        # Токенизация и POS-тегирование
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Определяем грамматику для извлечения фраз
        grammar = r"""
            PHRASE: {<JJ.*>*<NN.*>+}          # Прилагательные + существительные
                   {<NN.*><IN><NN.*>}         # Существительное + предлог + существительное
                   {<VB.*><NN.*>+}            # Глагол + существительные
        """
        
        # Создаем парсер и извлекаем фразы
        chunk_parser = nltk.RegexpParser(grammar)
        tree = chunk_parser.parse(tagged)
        
        phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'PHRASE'):
            phrase = ' '.join([word for word, tag in subtree.leaves()])
            if len(phrase.split()) > 1:  # Только фразы из нескольких слов
                phrases.append(phrase.lower())
        
        # Подсчитываем частоту фраз и берем топ-10
        from collections import Counter
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(10)]
        
    except Exception as e:
        logger.error(f"Error extracting phrases: {str(e)}")
        return []

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

def analyze_chunks(text_or_dict, chunk_size=500, overlap=50, progress_callback=None) -> list:
    """
    Анализирует чанки и возвращает подробную информацию о каждом
    """
    chunks = split_into_chunks(text_or_dict, chunk_size, overlap)
    total_chunks = len(chunks)
    
    def analyze_single_chunk(chunk_data):
        i, chunk = chunk_data
        progress = (i + 1) / total_chunks * 100
        
        # Базовая информация
        info = {
            'chunk_id': i,
            'length': len(chunk),
            'word_count': len(chunk.split()),
            'sentences': len(nltk.sent_tokenize(chunk)),
            'start': chunk[:50] + '...',
            'end': '...' + chunk[-50:],
        }
        
        # Извлечение метаданных
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'entities': executor.submit(extract_named_entities, chunk),
                'key_phrases': executor.submit(extract_key_phrases, chunk),
                'dates': executor.submit(extract_dates, chunk)
            }
            
            # Собираем результаты
            for key, future in futures.items():
                info[key] = future.result()
        
        if progress_callback:
            progress_callback({
                'stage': 'analyzing_chunks',
                'progress': progress,
                'current': i + 1,
                'total': total_chunks,
                'message': f'Analyzing chunk {i + 1}/{total_chunks}'
            })
        
        return info

    # Создаем список кортежей (индекс, чанк)
    chunk_data = list(enumerate(chunks))
    
    # Используем ThreadPoolExecutor для параллельной обработки чанков
    with ThreadPoolExecutor(max_workers=min(8, total_chunks)) as executor:
        chunks_info = list(tqdm(
            executor.map(analyze_single_chunk, chunk_data),
            total=total_chunks,
            desc="Analyzing chunks",
            unit="chunk"
        ))
    
    return chunks_info

def print_chunks_analysis(text_or_dict, chunk_size=500, overlap=50, progress_callback=None):
    """
    Асинхронно выводит подробный анализ чанков
    """
    logger.info("Starting chunks analysis...")
    chunks_info = analyze_chunks(text_or_dict, chunk_size, overlap, progress_callback)
    
    # Выводим результаты
    for info in chunks_info:
        print(f"\n{'='*80}")
        print(f"Chunk #{info['chunk_id']}")
        print(f"{'='*80}")
        print(f"Length: {info['length']} characters")
        print(f"Words: {info['word_count']}")
        print(f"Sentences: {info['sentences']}")
        print("\nStart:")
        print(info['start'])
        print("\nEnd:")
        print(info['end'])
        print("\nEntities:", ', '.join(info['entities']) if info['entities'] else 'None')
        print("\nKey phrases:", ', '.join(info['key_phrases']) if info['key_phrases'] else 'None')
        print("\nDates:", ', '.join(info['dates']) if info['dates'] else 'None')
    
    logger.info("Chunks analysis completed")