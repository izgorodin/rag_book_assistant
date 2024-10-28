from abc import ABC, abstractmethod
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
from typing import List, Union
import httpx
from src.embedding import EmbeddingService
from src.utils.logger import setup_logger

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
    def __init__(self, api_key: str = OPENAI_API_KEY):
        super().__init__(api_key)
        self.embedding_service = None  # Will be injected

    def set_embedding_service(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def generate_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are an AI assistant specialized in accurately extracting information from the provided text. "
            "Use only the information given in the context and avoid adding any external data. "
            "Provide answers using Markdown formatting for better readability, including lists, headings, and text highlighting where appropriate."
            "If the question requires detailed information, such as listing all relevant points or quoting specific passages, "
            "adjust the length of your response accordingly to fully address the request. "
            "If the required information is not available in the context, briefly explain what is known and mention that the specific information is missing. "
            "For example, if asked to list all key events, provide a comprehensive list based on the context. If asked for a quote, include the exact text if it is present in the context."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Please provide an answer based on the above context."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
        if not self.embedding_service:
            raise ValueError("EmbeddingService not initialized")
        return self.embedding_service.create_embeddings(texts)

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
