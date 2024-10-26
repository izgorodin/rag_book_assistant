"""
OpenAI Service Module

This module provides interfaces and implementations for interacting with OpenAI's API,
including text generation and embedding creation.
"""

from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List, Dict, Optional
from src.config import OPENAI_CONFIG
from src.logger import setup_logger
from src.error_handler import (
    handle_rag_error, OpenAIError
)
from src.types import (
    Embedding, EmbeddingInput, EmbeddingList, QueryType,
    ModelResponse, EmbeddingResponse, ChatMessage,
    ChatMessages, APIKey, ModelName, MaxTokens,
    ServiceResponse
)

logger = setup_logger()

class BaseOpenAIService(ABC):
    """Abstract base class defining the interface for OpenAI services."""
    
    @abstractmethod
    def generate_answer(self, query: QueryType, context: str) -> ModelResponse:
        """Generate an answer based on the query and context."""
        pass

    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> EmbeddingList:
        """Create embeddings for a list of texts."""
        pass

    @abstractmethod
    def create_embedding(self, text: str) -> Embedding:
        """Create embedding for a single text."""
        pass

class OpenAIService(BaseOpenAIService):
    """Implementation of OpenAI service for text generation and embeddings."""

    def __init__(self, api_key: APIKey):
        """Initialize OpenAI service."""
        self.client = OpenAI(api_key=OPENAI_CONFIG['api_key'])

    @handle_rag_error
    def generate_answer(self, query: QueryType, context: str) -> ModelResponse:
        """Generate an answer using OpenAI's chat completion."""
        messages: ChatMessages = [
            ChatMessage({
                "role": "system",
                "content": "You are a helpful assistant specialized in extracting precise information from texts."
            }),
            ChatMessage({
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            })
        ]
        
        try:
            response: ServiceResponse = self.client.chat.completions.create(
                model=ModelName(OPENAI_CONFIG['gpt_model']),
                messages=messages,
                max_tokens=MaxTokens(OPENAI_CONFIG['max_tokens'])
            )
            return ModelResponse(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            raise OpenAIError(f"Failed to generate answer: {str(e)}")

    @handle_rag_error
    def create_embeddings(self, texts: List[str]) -> EmbeddingList:
        """Create embeddings for multiple texts."""
        try:
            response: EmbeddingResponse = self.client.embeddings.create(
                input=texts,
                model=ModelName(OPENAI_CONFIG['embedding_model'])
            )
            return EmbeddingList([Embedding(embedding.embedding) for embedding in response.data])
        except Exception as e:
            logger.error(f"Error in create_embeddings: {str(e)}")
            raise OpenAIError(f"Failed to create embeddings: {str(e)}")

    @handle_rag_error
    def create_embedding(self, text: EmbeddingInput) -> Embedding:
        """Create embedding for a single text."""
        try:
            logger.info(f"Creating embedding for text: {text[:50]}...")
            response: EmbeddingResponse = self.client.embeddings.create(
                input=text,
                model=ModelName(OPENAI_CONFIG['embedding_model'])
            )
            return Embedding(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise OpenAIError(f"Error creating embedding: {str(e)}")
