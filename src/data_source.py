from abc import ABC, abstractmethod
from typing import List, Any

class DataSource(ABC):
    @abstractmethod
    def get_chunks(self) -> List[str]:
        pass

    @abstractmethod
    def get_embeddings(self) -> List[List[float]]:
        pass

    @abstractmethod
    def get_processed_text(self) -> Any:
        pass