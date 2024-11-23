from abc import ABC, abstractmethod
from typing import Optional

class LLMInterface(ABC):
    """Abstract base class for Language Model services."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response from the language model.
        
        Args:
            prompt: The input prompt for the model
            temperature: Controls randomness in the response (0.0 to 1.0)
            
        Returns:
            Generated response as string
        """
        pass
