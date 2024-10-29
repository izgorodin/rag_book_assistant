from typing import Union, Optional
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
import httpx
from src.utils.logger import get_main_logger

logger = get_main_logger()

class OpenAIErrorFactory:
    """Фабрика для создания ошибок OpenAI API."""
    
    @staticmethod
    def create_error(
        error_type: type,
        message: str = "Test error",
        status_code: int = 400
    ) -> Union[RateLimitError, APIError, APITimeoutError, APIConnectionError]:
        """
        Создает ошибку OpenAI API заданного типа.
        
        Args:
            error_type: Тип ошибки OpenAI
            message: Сообщение об ошибке
            status_code: HTTP код ошибки
        """
        # Создаем базовые объекты для ошибок
        request = httpx.Request("GET", "https://api.openai.com/v1/test")
        response = httpx.Response(status_code=status_code, request=request)
        
        logger.debug(f"Creating OpenAI error: {error_type.__name__}")
        
        if error_type == RateLimitError:
            return RateLimitError(
                message=message,
                response=response,
                body={"error": {"message": message}}
            )
        elif error_type == APIError:
            return APIError(
                message=message,
                request=request,
                body={"error": {"message": message}}
            )
        elif error_type == APITimeoutError:
            return APITimeoutError(request=request)
        elif error_type == APIConnectionError:
            return APIConnectionError(
                message=message,
                request=request
            )
        else:
            raise ValueError(f"Unsupported error type: {error_type}")