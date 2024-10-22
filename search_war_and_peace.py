import pickle
import os
import sys
import numpy as np
from typing import List, Dict, Any
from src.embedding import create_embeddings, cosine_similarity
from src.rag import rag_query

print("Script started")

class VectorSearch:
    def __init__(self, chunks: List[str], embeddings: List[List[float]]):
        print(f"Initializing VectorSearch with {len(chunks)} chunks and {len(embeddings)} embeddings")
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"Searching for query: '{query}'")
        query_embedding = create_embeddings([query])[0]
        scores = self._get_embedding_scores(query_embedding)
        return self._get_top_chunks(scores, top_k)

    def _get_embedding_scores(self, query_embedding: List[float]) -> np.ndarray:
        print("Calculating embedding scores")
        return np.array([cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in self.embeddings])

    def _get_top_chunks(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        print(f"Getting top {top_k} chunks")
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{'chunk': self.chunks[i], 'score': scores[i]} for i in top_indices]

def load_embeddings(file_path):
    print(f"Attempting to load embeddings from {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data loaded. Type: {type(data)}")
        
        if isinstance(data, tuple) and len(data) == 2:
            chunks, embeddings = data
            print("Data is a tuple with 2 elements")
        elif isinstance(data, dict):
            chunks = data['chunks']
            embeddings = data['embeddings']
            print("Data is a dictionary")
        else:
            print(f"Error: Unexpected data format in {file_path}")
            return None, None
        
        print(f"Successfully loaded {len(chunks)} chunks and {len(embeddings)} embeddings.")
        return chunks, embeddings
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None, None

def main():
    print("Starting the War and Peace vector search script")
    
    # Загрузка эмбеддингов
    embeddings_file = 'data/embeddings/voina-i-mir_chunks_embeddings.pkl'
    chunks, embeddings = load_embeddings(embeddings_file)
    
    if chunks is None or embeddings is None:
        print("Failed to load embeddings. Exiting.")
        sys.exit(1)
    
    print("Creating VectorSearch instance")
    try:
        vector_search = VectorSearch(chunks, embeddings)
        print("VectorSearch instance created successfully")
    except Exception as e:
        print(f"Error creating VectorSearch instance: {str(e)}")
        sys.exit(1)

    print("\nReady for questions. Type 'exit' to quit.")
    
    # Интерактивный режим для запросов
    while True:
        query = input("\nВведите ваш вопрос: ")
        if query.lower() == 'exit':
            break

        try:
            results = vector_search.search(query)
            print("\nТоп 5 наиболее релевантных отрывков:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Оценка: {result['score']:.4f}")
                print(f"Отрывок: {result['chunk'][:200]}...")  # Показываем первые 200 символов
        except Exception as e:
            print(f"Error searching: {str(e)}")

    print("Exiting the script")

if __name__ == "__main__":
    main()
