from abc import abstractmethod, ABC
from contextlib import contextmanager
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Callable, Generator
from src.config import PINECONE_API_KEY, PINECONE_CLOUD, EMBEDDING_DIMENSION, PINECONE_INDEX_NAME, PINECONE_METRIC, PINECONE_REGION
from src.cache_manager import get_cache_key, save_to_cache, load_from_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from src.logger import setup_logger    

logger = setup_logger()


class PineconeInterface(ABC):
    @abstractmethod
    def list_indexes(self):
        pass

    @abstractmethod
    def create_index(self, name: str, dimension: int, metric: str, spec: Any):
        pass

    @abstractmethod
    def Index(self, name: str):
        pass

class BasePineconeManager:
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def upsert_embeddings(self, chunks: List[str], embeddings: List[List[float]]):
        pass

    @abstractmethod
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_index(self):
        pass

    @abstractmethod
    def get_or_create_embeddings(self, chunks: List[str], embedding_function) -> List[List[float]]:
        pass

    @abstractmethod
    def batch_operation(self) -> Generator[None, None, None]:
        pass

class PineconeManager(BasePineconeManager):
    def __init__(self, index_name: str = PINECONE_INDEX_NAME, pinecone_client: Pinecone = None,
                 max_retries: int = 3, min_wait: int = 1, max_wait: int = 10):
        self.index_name = index_name
        self.pc = pinecone_client or Pinecone(api_key=PINECONE_API_KEY)
        self.index = None
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        self._initialize_index()

    def _initialize_index(self):
        @retry(stop=stop_after_attempt(self.max_retries),
               wait=wait_exponential(multiplier=1, min=self.min_wait, max=self.max_wait),
               retry=retry_if_exception_type((Exception,)),
               reraise=True)
        def _initialize():
            try:
                indexes = self.pc.list_indexes()
                if self.index_name not in indexes:
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=EMBEDDING_DIMENSION,
                        metric=PINECONE_METRIC,
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
                    )
                else:
                    logger.info(f"Index {self.index_name} already exists. Using existing index.")
                
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Successfully initialized Pinecone index: {self.index_name}")
            except PineconeApiException as e:
                if "ALREADY_EXISTS" in str(e):
                    logger.info(f"Index {self.index_name} already exists. Using existing index.")
                    self.index = self.pc.Index(self.index_name)
                else:
                    logger.error(f"Error initializing Pinecone: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error initializing Pinecone: {str(e)}")
                raise

        _initialize()

    def _pinecone_query(self, vector: List[float], top_k: int, include_metadata: bool = True) -> Dict[str, Any]:
        try:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
        except ValueError as e:
            if "argument order" in str(e):
                # Fallback to old API if the new one fails
                return self.index.query(vector, top_k=top_k, include_metadata=include_metadata)
            raise

    def is_available(self) -> bool:
        return self.index is not None

    def upsert_embeddings(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        if not self.is_available():
            raise ValueError("Pinecone index is not initialized")
        
        vectors = [(str(i), embedding, {"chunk": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        self.index.upsert(vectors=vectors)

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise ValueError("Pinecone index is not initialized")
        
        results = self._pinecone_query(query_embedding, top_k=top_k, include_metadata=True)
        return [{"chunk": match['metadata'].get('chunk', ''), "score": match['score']} for match in results['matches']]

    def clear_index(self) -> None:
        if self.is_available():
            self.index.delete(delete_all=True)
        else:
            logger.warning("Pinecone index is not available. Skipping clear operation.")

    def get_or_create_embeddings(self, chunks: List[str], embedding_function: Callable[[List[str]], List[List[float]]]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        new_chunks: List[str] = []
        new_chunk_indices: List[int] = []

        for i, chunk in enumerate(chunks):
            cache_key = get_cache_key(chunk)
            cached_embedding = load_from_cache(cache_key)
            if cached_embedding is not None:
                all_embeddings.append(cached_embedding)
            else:
                new_chunks.append(chunk)
                new_chunk_indices.append(i)

        if new_chunks:
            new_embeddings = embedding_function(new_chunks)
            for chunk, emb, index in zip(new_chunks, new_embeddings, new_chunk_indices):
                cache_key = get_cache_key(chunk)
                save_to_cache(cache_key, emb)
                all_embeddings.insert(index, emb)

        return all_embeddings

    @contextmanager
    def batch_operation(self) -> Generator[None, None, None]:
        # This is a placeholder for potential batch operations
        # In a real implementation, you might start a transaction or prepare a batch
        yield
        # After yield, you might commit the transaction or execute the batch

    def check_pinecone_index(self):
        try:
            index_stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index stats: {index_stats}")
            return index_stats
        except Exception as e:
            logger.error(f"Error checking Pinecone index: {str(e)}", exc_info=True)
            return None
