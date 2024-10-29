from abc import ABC, abstractmethod
from typing import Any, List  # Import abstract base class and abstract method for interface definition

class DataSource(ABC):
    @abstractmethod
    def get_chunks(self) -> List[str]:
        """Retrieve the text chunks from the data source."""
        pass

    @abstractmethod
    def get_embeddings(self) -> List[List[float]]:
        """Retrieve the embeddings corresponding to the text chunks."""
        pass

    @abstractmethod
    def get_processed_text(self) -> Any:
        """Retrieve the processed text data."""
        pass
