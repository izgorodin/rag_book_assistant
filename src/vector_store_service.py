from typing import List, Dict, Any
from logger import setup_logger
from pinecone_manager import PineconeManager

logger = setup_logger()

class VectorStoreService:
    def __init__(self, vector_store: PineconeManager):
        self.vector_store = vector_store
        
    def store_vectors(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Store vectors in the vector store."""
        vectors = [
            {
                'id': str(i),
                'values': emb,
                'metadata': {'text': chunk}
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        
        if not self.vector_store.is_available():
            raise ValueError("Vector store is not available")
            
        try:
            self.vector_store.upsert_vectors(vectors)
            logger.info(f"Successfully stored {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise
