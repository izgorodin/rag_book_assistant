from typing import List, Dict, Any, Optional  # Import necessary types for type hinting
import pickle  # Import pickle for object serialization
import os  # Import os for file and directory operations
from src.data_source import DataSource
from src.embedding import EmbeddingService  # Import the EmbeddingService for embedding functionalities
from src.vector_store_service import VectorStoreService  # Import the VectorStoreService for vector store functionalities
from src.search import get_search_strategy
import re  # Import re for regular expressions
from src.utils.logger import get_main_logger, get_rag_logger
import nltk
from nltk.tokenize import word_tokenize
import asyncio


class BookDataInterface(DataSource):
    def __init__(self, 
                 chunks: List[str],  # List of text chunks
                 embeddings: List[List[float]],  # List of embeddings for the chunks
                 processed_text: Dict[str, Any],  # Dictionary containing processed text data
                 embedding_service: EmbeddingService,  # Instance of EmbeddingService for embedding operations
                 vector_store_service: VectorStoreService,  # Instance of VectorStoreService for vector store operations
                 dates: Optional[List[str]] = None,  # Optional list of dates associated with the chunks
                 entities: Optional[List[Dict[str, Any]]] = None,  # Optional list of entities found in the text
                 key_phrases: Optional[List[str]] = None):  # Optional list of key phrases extracted from the text
        self._chunks = chunks  # Initialize the chunks
        self._embeddings = embeddings  # Initialize the embeddings
        self._processed_text = processed_text  # Initialize the processed text
        self._embedding_service = embedding_service  # Initialize the embedding service
        self._vector_store_service = vector_store_service  # Initialize the vector store service
        self._dates = dates or []  # Initialize dates, default to empty list if None
        self._entities = entities or []  # Initialize entities, default to empty list if None
        self._key_phrases = key_phrases or []  # Initialize key phrases, default to empty list if None
        self.logger = get_main_logger()
        self.rag_logger = get_rag_logger()
        
    @classmethod
    def from_file(cls, file_path: str):
        """Create an instance of BookDataInterface from a file."""
        with open(file_path, 'rb') as f:  # Open the file in binary read mode
            data = pickle.load(f)  # Load the data from the file
        return cls(data['chunks'], data['embeddings'], data.get('processed_text', {}), 
                   data.get('embedding_service', {}), data.get('vector_store_service', {}), data.get('dates', []), 
                   data.get('entities', []), data.get('key_phrases', []))  # Return an instance with loaded data

    def save(self, file_path: str):
        """Save the current instance data to a file."""
        data = {
            'chunks': self._chunks,  # Store chunks
            'embeddings': self._embeddings,  # Store embeddings
            'processed_text': self._processed_text,  # Store processed text
            'dates': self._dates,  # Store dates
            'entities': self._entities,  # Store entities
            'key_phrases': self._key_phrases  # Store key phrases
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        with open(file_path, 'wb') as f:  # Open the file in binary write mode
            pickle.dump(data, f)  # Serialize and save the data

    def __len__(self):
        """Return the number of chunks."""
        return len(self._chunks)  # Return the length of the chunks list

    def get_chunks(self) -> List[str]:
        """Return the list of text chunks."""
        return self._chunks  # Return the stored chunks

    def get_embeddings(self) -> List[List[float]]:
        """Return the list of embeddings."""
        return self._embeddings  # Return the stored embeddings

    def get_processed_text(self) -> Dict[str, Any]:
        """Return the processed text data."""
        return self._processed_text  # Return the processed text

    def get_dates(self) -> List[str]:
        """Return the list of dates associated with the chunks."""
        return self._dates  # Return the stored dates

    def get_entities(self) -> List[Dict[str, Any]]:
        """Return the list of entities found in the text."""
        return self._entities  # Return the stored entities

    def get_key_phrases(self) -> List[str]:
        """Return the list of key phrases extracted from the text."""
        return self._key_phrases  # Return the stored key phrases

    async def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Get the most relevant chunks for a given query."""
        try:
            # Создаем эмбеддинги для запроса
            embeddings = await self._embedding_service.create_embeddings([query])
            query_embedding = embeddings[0]
            
            # Определяем тип запроса через NLP анализ
            query_analysis = await self._analyze_query(query)
            
            # Применяем соответствующую стратегию поиска
            if query_analysis['is_factual']:
                filter_conditions = await self._build_filter_conditions(query_analysis)
                results = await self._vector_store_service.search_vectors(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter_conditions=filter_conditions
                )
            else:
                results = await self._vector_store_service.search_vectors(
                    query_vector=query_embedding,
                    top_k=top_k
                )
            
            if results and isinstance(results, list):
                return [result.get("metadata", {}).get("text", "") for result in results]
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting relevant chunks: {str(e)}")
            return []

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query type and extract key information asynchronously."""
        # Определяем язык запроса через NLTK асинхронно
        tokens = await asyncio.to_thread(word_tokenize, query.lower())
        
        # Определяем язык по наличию кириллицы
        has_cyrillic = bool(re.search('[а-яА-Я]', query))
        lang = 'ru' if has_cyrillic else 'en'
        
        # Паттерны для разных языков
        patterns = {
            'en': {
                'factual': [
                    r'\b(who|what|where|when|why|how)\b',
                    r'\b(date|year|time|place|location)\b',
                    r'\b(person|people|name)\b'
                ]
            },
            'ru': {
                'factual': [
                    r'\b(кто|что|где|когда|почему|как)\b',
                    r'\b(дата|год|время|место|локация)\b',
                    r'\b(человек|люди|имя)\b'
                ]
            }
        }
        
        # Выбираем паттерны в зависимости от языка
        lang_patterns = patterns.get(lang, patterns['en'])
        
        # Проверяем на фактические паттерны асинхронно
        is_factual = any(
            bool(re.search(pattern, query.lower()))
            for pattern in lang_patterns['factual']
        )
        
        return {
            'is_factual': is_factual,
            'language': lang,
            'contains_date': bool(re.search(r'\b\d{4}\b', query)),
            'contains_name': bool(re.search(r'\b[A-Z][a-z]+\b', query))
        }

    async def _build_filter_conditions(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter conditions based on query analysis asynchronously."""
        conditions = {"$or": []}
        
        if query_analysis['contains_date']:
            conditions["$or"].append({"has_date": True})
        
        if query_analysis['contains_name']:
            conditions["$or"].append({"has_names": True})
        
        if not conditions["$or"]:
            conditions["$or"] = [
                {"has_date": True},
                {"has_year": True},
                {"has_names": True}
            ]
        
        return conditions
