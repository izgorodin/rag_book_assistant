from typing import List, Dict, Any
import pickle
import os
from src.data_source import DataSource
from src.embedding import EmbeddingService
from src.services.llm_interface import LLMInterface
from src.vector_store_service import VectorStoreService
from src.search import get_search_strategy
from src.utils.logger import get_main_logger, get_rag_logger
from src.services.query_processor import QueryProcessor

class BookDataInterface(DataSource):
    def __init__(self, 
                 namespace: str,
                 chunks: List[str],
                 embeddings: List[List[float]],
                 processed_text: Dict[str, Any],
                 embedding_service: EmbeddingService,
                 vector_store_service: VectorStoreService,
                 llm_service: LLMInterface,
                 metadata: Dict[str, Any]):
        self.namespace = namespace
        self._chunks = chunks
        self._embeddings = embeddings
        self._processed_text = processed_text
        self._embedding_service = embedding_service
        self._vector_store_service = vector_store_service
        self._llm_service = llm_service
        self._metadata = metadata
        self.logger = get_main_logger()
        self.query_processor = QueryProcessor(llm_service)

    # Реализация DataSource
    def get_chunks(self) -> List[str]:
        return self._chunks
        
    def get_embeddings(self) -> List[List[float]]:
        return self._embeddings
        
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'total_chunks': len(self._chunks),
            'key_entities': self.entities,
            'key_phrases': self.key_phrases,
            'dates': self.dates,
            **self._metadata
        }

    async def search(self, 
                    query: str, 
                    conversation_history: List[Dict[str, str]] = None,
                    search_strategy: str = "cosine") -> List[Dict[str, Any]]:
        """Выполняет поиск используя выбранную стратегию"""
        self.logger.info(f"Starting search for query: {query}")
        
        # Подготавливаем запрос через QueryProcessor
        prepared_query = await self.query_processor.prepare_search_query(
            query,
            conversation_history,
            self.get_metadata()
        )
        
        # Получаем и используем стратегию поиска
        strategy = get_search_strategy(
            search_strategy,
            self,
            self._embedding_service
        )
        
        # Выполняем поиск
        results = await strategy.search(
            prepared_query['enhanced_query'],
            prepared_query['search_params']['top_k']
        )
        
        self.logger.info(f"Found {len(results)} relevant chunks")
        return results
