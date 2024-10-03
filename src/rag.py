from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)

def find_most_relevant_chunks(query: str, query_embedding: List[float], chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS
    )
    
    return response.choices[0].message.content

def rag_query(query: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    query_embedding = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    relevant_chunks = find_most_relevant_chunks(query, query_embedding, chunks, embeddings)
    context = " ".join(relevant_chunks)
    
    prompt = f"""
    Text: {context}

    Based solely on the information provided in the text above, please answer the following question.
    If the text doesn't contain enough information to answer the question, say "I don't have enough information to answer that question."

    Question: {query}

    Answer:
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the given information."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()