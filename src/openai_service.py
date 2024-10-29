from abc import ABC, abstractmethod
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
from typing import List, Union
import httpx
from src.embedding import EmbeddingService
from src.utils.logger import get_main_logger, get_rag_logger

# Initialize loggers for main and RAG (Retrieval-Augmented Generation) processes
logger = get_main_logger()
rag_logger = get_rag_logger()

class BaseOpenAIService(ABC):
    """
    Abstract base class for OpenAI services.
    Defines the interface for generating answers and creating embeddings.
    """

    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initializes the OpenAI client with the provided API key.

        Args:
            api_key (str): The API key for OpenAI.
        """
        self.client = OpenAI(api_key=api_key)

    @abstractmethod
    def generate_answer(self, query: str, context: str) -> str:
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
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
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

    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initializes the OpenAI service with the provided API key.

        Args:
            api_key (str): The API key for OpenAI.
        """
        super().__init__(api_key)
        self.embedding_service = None  # Will be injected

    def set_embedding_service(self, embedding_service: EmbeddingService):
        """
        Set the embedding service to be used for creating embeddings.

        Args:
            embedding_service (EmbeddingService): The embedding service instance.
        """
        self.embedding_service = embedding_service

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on the provided query and context.

        Args:
            query (str): The question to be answered.
            context (str): The context from which to extract information.

        Returns:
            str: The generated answer.
        """
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
        rag_logger.info(
            f"\nQuery Processing:\n"
            f"Query: {query}\n"
            f"Context length: {len(context)} chars\n"
            f"{'-'*50}"
        )
        
        try:
            # Call the OpenAI API to generate a chat completion
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            answer = response.choices[0].message.content
            
            rag_logger.info(
                f"\nGenerated Answer:\n"
                f"Answer length: {len(answer)} chars\n"
                f"Model: {GPT_MODEL}\n"
                f"{'-'*50}"
            )
            
            return answer
        except Exception as e:
            # Handle any exceptions that occur during the API call
            error = self._handle_openai_error(e)
            error_msg = f"Error in generate_answer: {str(error)}"
            logger.error(error_msg)
            rag_logger.error(f"\nOpenAI Error:\n{error_msg}\n{'-'*50}")
            return f"Sorry, I encountered an error while generating the answer: {str(error)}"

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
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
        return self.embedding_service.create_embeddings(texts)

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
