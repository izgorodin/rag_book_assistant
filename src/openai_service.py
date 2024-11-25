from abc import ABC, abstractmethod
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
from typing import List, Union
import httpx
from src.embedding import EmbeddingService
from src.utils.logger import get_main_logger, get_rag_logger
import openai

# Initialize loggers for main and RAG (Retrieval-Augmented Generation) processes
logger = get_main_logger()
rag_logger = get_rag_logger()

class BaseOpenAIService(ABC):
    """
    Abstract base class for OpenAI services.
    Defines the interface for generating answers and creating embeddings.
    """

    def __init__(self, client):
        """
        Initializes with provided OpenAI client.
        """
        self.client = client

    @abstractmethod
    async def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on the provided query and context.

        Args:
            query (str): The question to be answered.
            context (str): The context from which to extract information.

        Returns:
            str: The generated answer.
        """
        pass

    @abstractmethod
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for the provided texts.

        Args:
            texts (List[str]): A list of texts to create embeddings for.

        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.
        """
        pass

    @abstractmethod
    def _handle_openai_error(self, error: Exception) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        """
        Handle errors from OpenAI API calls.

        Args:
            error (Exception): The exception raised during the API call.

        Returns:
            Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]: The handled error.
        """
        pass

    @abstractmethod
    def create_test_exception(self, error_type: type, message: str) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        """
        Create a test exception for the specified error type.

        Args:
            error_type (type): The type of error to create.
            message (str): The error message.

        Returns:
            Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]: The created exception.
        """
        pass

class OpenAIService(BaseOpenAIService):
    """
    Implementation of the OpenAI service that generates answers and creates embeddings.
    """

    def __init__(self, client=None):
        """
        Initializes with provided OpenAI client.
        
        Args:
            client: Optional pre-configured OpenAI client. If None, creates new AsyncOpenAI client.
        """
        if client is None:
            logger.info(f"Creating new AsyncOpenAI client. OpenAI version: {openai.__version__}")
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        super().__init__(client)
        self.embedding_service = None
        logger.debug(f"Client structure: {dir(self.client)}")
        logger.debug(f"Chat structure: {dir(self.client.chat)}")
        logger.debug(f"Completions structure: {dir(self.client.chat.completions)}")

    def set_embedding_service(self, embedding_service: EmbeddingService):
        """
        Set the embedding service to be used for creating embeddings.

        Args:
            embedding_service (EmbeddingService): The embedding service instance.
        """
        self.embedding_service = embedding_service

    async def generate_answer(self, query: str, context: str) -> str:
        try:
            logger.info(f"Generating answer for query: {query}")
            logger.debug(f"Using OpenAI client: {self.client}")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._format_user_prompt(query, context)}
            ]
            
            logger.info(f"Generating answer for query: {query}")
            
            # В v1 API используется просто create, но возвращает уже готовый объект
            response = await self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            
            # Получаем контент напрямую из объекта
            answer = response.choices[0].message.content
            self._log_response(answer)
            return answer
            
        except Exception as e:
            error = self._handle_openai_error(e)
            error_msg = f"Error in generate_answer: {str(error)}"
            logger.error(error_msg)
            return f"Sorry, I encountered an error: {str(error)}"

    def _get_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in accurately extracting information from the provided text. "
            "Use only the information given in the context and avoid adding any external data. "
            "If the required information is not available in the context, briefly explain what is known "
            "and mention that the specific information is missing."
        )

    def _format_user_prompt(self, query: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Please provide an answer based on the above context."
        )

    def _log_response(self, answer: str) -> None:
        rag_logger.info(
            f"\nGenerated Answer:\n"
            f"Answer length: {len(answer)} chars\n"
            f"Model: {GPT_MODEL}\n"
            f"{'-'*50}"
        )

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for the provided texts.

        Args:
            texts (List[str]): A list of texts to create embeddings for.

        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.

        Raises:
            ValueError: If the embedding service is not initialized.
        """
        if not self.embedding_service:
            raise ValueError("EmbeddingService not initialized")
        
        return await self.embedding_service.create_embeddings(texts)

    def _handle_openai_error(self, error: Exception) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        """
        Handle errors from OpenAI API calls.

        Args:
            error (Exception): The exception raised during the API call.

        Returns:
            Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]: The handled error.
        """
        if isinstance(error, (RateLimitError, APIError, APITimeoutError, APIConnectionError)):
            return error
        else:
            return Exception(f"An unexpected error occurred: {str(error)}")

    @staticmethod
    def create_test_exception(error_type: type, message: str) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]:
        """
        Create a test exception for the specified error type.

        Args:
            error_type (type): The type of error to create.
            message (str): The error message.

        Returns:
            Union[RateLimitError, APIError, APITimeoutError, APIConnectionError, Exception]: The created exception.
        """
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
