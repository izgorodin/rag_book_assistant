from abc import ABC, abstractmethod
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
from typing import List, Dict, Union
import httpx
from src.logger import setup_logger

logger = setup_logger()



class BaseOpenAIService(ABC):
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)

    @abstractmethod
    def generate_answer(self, query: str, context: str) -> str:
        pass

    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def _handle_openai_error(self, error: Exception) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        pass

    @abstractmethod
    def create_test_exception(self, error_type: type, message: str) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        pass

class OpenAIService(BaseOpenAIService):
    def generate_answer(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in extracting precise information from texts. Focus on providing accurate information. If the exact information is not available, explain what is known and what is missing."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nProvide a concise answer based on the context. If specific information is not available, briefly explain what is known and what is missing."}
        ]
        
        logger.info(f"Generating answer for query: {query}")
        
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            error = self._handle_openai_error(e)
            logger.error(f"Error in generate_answer: {str(error)}")
            return f"Sorry, I encountered an error while generating the answer: {str(error)}"

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(input=texts, model="text-embedding-ada-002")
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            error = self._handle_openai_error(e)
            logger.error(f"Error in create_embeddings: {str(error)}")
            raise ValueError(f"Failed to create embeddings: {str(error)}")

    def _handle_openai_error(self, error: Exception) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        if isinstance(error, (RateLimitError, APIError, APITimeoutError, APIConnectionError)):
            return error
        else:
            return Exception(f"An unexpected error occurred: {str(error)}")

    @staticmethod
    def create_test_exception(error_type: type, message: str) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        mock_response = httpx.Response(status_code=400, request=httpx.Request("GET", "https://api.openai.com/v1/test"))
        if error_type == RateLimitError:
            return RateLimitError(message=message, response=mock_response, body={})
        elif error_type == APIError:
            return APIError(message=message, request=mock_response.request, body={})
        elif error_type == APITimeoutError:
            return APITimeoutError(request=mock_response.request)
        elif error_type == APIConnectionError:
            return APIConnectionError(message=message, request=mock_response.request)
        else:
            return Exception(message)
