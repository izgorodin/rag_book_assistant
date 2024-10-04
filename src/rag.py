from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def find_most_relevant_chunks(query: str, query_embedding: List[float], chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. Be sure to mention key terms from the context in your answer."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

def rag_query(query: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        query_embedding = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
        relevant_chunks = find_most_relevant_chunks(query, query_embedding, chunks, embeddings)
        context = " ".join(relevant_chunks)
        full_context = f"Original text: {context}\n\nQuestion: {query}"
        answer = generate_answer(query, full_context)
        logger.info(f"Successfully generated answer for RAG query: {query}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"