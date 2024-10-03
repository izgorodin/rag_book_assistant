from typing import List
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def create_embeddings(chunks: List[str]) -> List[List[float]]:
       embeddings = []
       for chunk in chunks:
           try:
               response = client.embeddings.create(
                   input=chunk,
                   model=EMBEDDING_MODEL
               )
               embeddings.append(response.data[0].embedding)
           except Exception as e:
               raise Exception(f"Error creating embedding: {str(e)}")
       return embeddings
   