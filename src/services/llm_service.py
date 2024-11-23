from src.openai_service import OpenAIService
from src.services.llm_interface import LLMInterface
from src.utils.logger import get_main_logger, get_rag_logger

logger = get_main_logger()
rag_logger = get_rag_logger()

class LLMService(LLMInterface):
    """LLM service that uses OpenAI implementation."""
    
    def __init__(self, client):
        """Initialize with OpenAI client."""
        self.openai_service = OpenAIService(client)
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate a response using OpenAI service."""
        return self.openai_service.generate_answer(query, context)
