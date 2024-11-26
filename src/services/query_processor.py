from typing import Dict, List, Any
from src.config import SEARCH_SETTINGS
from src.utils.logger import get_main_logger

class QueryProcessor:
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.logger = get_main_logger()
        self.default_settings = SEARCH_SETTINGS
        
    async def prepare_search_query(self, 
                                 query: str, 
                                 conversation_history: List[Dict[str, str]] = None,
                                 document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        self.logger.info(f"Preparing search query. Original query: {query}")
        self.logger.debug(f"Context: {len(conversation_history) if conversation_history else 0} messages")
        self.logger.debug(f"Document metadata: {document_metadata}")
        
        try:
            prepared_query = await self._prepare_with_llm(
                query, 
                conversation_history, 
                document_metadata
            )
            self.logger.info(f"LLM enhanced query: {prepared_query.get('search_query')}")
            self.logger.info(f"Search parameters: top_k={prepared_query.get('top_k')}, threshold={prepared_query.get('threshold')}")
            
            return {
                'original_query': query,
                'enhanced_query': prepared_query.get('search_query', query),
                'search_params': {
                    'top_k': prepared_query.get('top_k', self.default_settings['top_k_chunks']),
                    'threshold': prepared_query.get('threshold', self.default_settings['similarity_threshold'])
                }
            }
        except Exception as e:
            self.logger.error(f"Error in query preparation: {e}", exc_info=True)
            return {
                'original_query': query,
                'enhanced_query': query,
                'search_params': {
                    'top_k': self.default_settings['top_k_chunks'],
                    'threshold': self.default_settings['similarity_threshold']
                }
            }
    
    def _build_preparation_prompt(self, 
                                query: str, 
                                conversation_history: List[Dict[str, str]] = None,
                                document_metadata: Dict[str, Any] = None) -> str:
        """Создает промпт для LLM."""
        prompt = [
            "You are a search query optimization expert. Your task is to enhance the search query",
            "for a vector database containing book content chunks.",
            "\nContext:",
            f"Original query: {query}"
        ]
        
        if conversation_history:
            prompt.append("\nRecent conversation context:")
            for msg in conversation_history[-3:]:
                prompt.append(f"- {msg['role']}: {msg['content']}")
        
        if document_metadata:
            prompt.append("\nDocument analysis:")
            prompt.append(f"- Total chunks: {document_metadata.get('total_chunks', 'unknown')}")
            prompt.append(f"- Key entities: {', '.join(document_metadata.get('key_entities', []))}")
            prompt.append(f"- Key phrases: {', '.join(document_metadata.get('key_phrases', []))}")
            prompt.append(f"- Dates mentioned: {', '.join(document_metadata.get('dates', []))}")
        
        prompt.extend([
            "\nBased on the above context, please provide:",
            "1. An enhanced search query that will help find the most relevant chunks",
            "2. Recommended number of chunks to retrieve (top_k) based on the query complexity",
            "3. Minimum similarity threshold (0.0-1.0) based on required precision",
            "\nRespond in JSON format:",
            "{",
            '  "search_query": "enhanced query text",',
            '  "top_k": number (1-10),',
            '  "threshold": float (0.5-0.95)',
            "}"
        ])
        
        return "\n".join(prompt) 
    
    async def _prepare_with_llm(self, 
                              query: str, 
                              conversation_history: List[Dict[str, str]] = None,
                              document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Подготавливает запрос с помощью LLM."""
        prompt = self._build_preparation_prompt(
            query,
            conversation_history,
            document_metadata
        )
        
        try:
            response = await self.llm_service.generate_text(prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error in LLM preparation: {e}")
            raise