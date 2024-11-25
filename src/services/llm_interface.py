from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """Abstract base class for Language Model services."""
    
    @abstractmethod
    async def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response from the language model.
        
        Args:
            query: The question to answer
            context: The context to use for answering
            
        Returns:
            Generated response as string
        """
        pass
