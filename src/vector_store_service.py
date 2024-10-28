from typing import List, Dict, Any
from src.utils.logger import setup_logger
from src.utils.error_handler import handle_rag_error
from src.pinecone_manager import PineconeManager

logger = setup_logger()

class VectorStoreService:
    def __init__(self, vector_store: PineconeManager):
        self.vector_store = vector_store
        self.max_batch_bytes = 4_000_000  # Pinecone limit: 4MB
        self.min_batch_size = 10  # Минимальный размер батча
        self.max_batch_size = 1000  # Максимальный размер батча

    def _calculate_batch_size(self, chunks: List[str], embeddings: List[List[float]]) -> int:
        """Calculate optimal batch size based on data size."""
        # Берем первые элементы для оценки размера
        sample_chunk = chunks[0]
        sample_embedding = embeddings[0]
        
        # Оцениваем размер одной записи (с запасом для метаданных)
        estimated_size = (
            len(sample_chunk.encode('utf-8')) +  # размер текста
            len(str(sample_embedding)) * 8 +     # размер эмбеддинга
            100                                  # доп. размер для метаданных
        )
        
        # Вычисляем оптимальный размер батча
        optimal_size = max(
            min(
                self.max_batch_bytes // estimated_size,  # размер по байтам
                self.max_batch_size                      # максимальный предел
            ),
            self.min_batch_size                         # минимальный предел
        )
        
        logger.info(f"Calculated optimal batch size: {optimal_size} " 
                   f"(estimated record size: {estimated_size} bytes)")
        return optimal_size

    @handle_rag_error
    def store_vectors(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Store vectors in batches respecting size limits."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        optimal_batch_size = self._calculate_batch_size(chunks, embeddings)
        total_items = len(chunks)
        
        logger.info(f"Storing {total_items} vectors with batch size {optimal_batch_size}")
        
        for i in range(0, total_items, optimal_batch_size):
            batch_end = min(i + optimal_batch_size, total_items)
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            # Создаем векторы для batch
            vectors = [
                {
                    'id': str(idx + i),
                    'values': emb,
                    'metadata': {'text': chunk}
                }
                for idx, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings))
            ]
            
            try:
                self.vector_store.upsert_vectors(vectors)
                logger.info(f"Stored batch {i//optimal_batch_size + 1}")
            except Exception as e:
                logger.error(f"Error storing batch {i//optimal_batch_size + 1}: {str(e)}")
                raise
