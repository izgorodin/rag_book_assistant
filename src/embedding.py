from typing import List
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# Separate function for testing
def create_embeddings_with_error_handling(chunks: List[str]) -> List[List[float]]:
    try:
        return create_embeddings(chunks)
    except Exception as e:
        raise Exception(f"Error creating embedding: {str(e)}")