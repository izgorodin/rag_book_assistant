from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def find_most_relevant_chunks(query: str, query_embedding: List[float], chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
    logger.debug(f"Finding most relevant chunks. Query: {query}")
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    logger.debug(f"Calculated similarities. Shape: {similarities.shape}")
    logger.debug(f"Max similarity: {np.max(similarities)}, Min similarity: {np.min(similarities)}")
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    logger.debug(f"Top {top_k} relevant chunk indices: {top_indices}")
    relevant_chunks = [chunks[i] for i in top_indices]
    for i, chunk in enumerate(relevant_chunks):
        logger.debug(f"Relevant chunk {i} (similarity: {similarities[top_indices[i]]:.4f}): {chunk[:100]}...")
    return relevant_chunks

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say that you don't know."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    logger.debug(f"Full context sent to GPT: {context}")
    logger.debug(f"Generating answer for query: {query}")
    logger.debug(f"Using GPT model: {GPT_MODEL}")
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        answer = response.choices[0].message.content
        logger.debug(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

def rag_query(query: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        logger.debug(f"Total chunks: {len(chunks)}, Total embeddings: {len(embeddings)}")
        logger.debug(f"First chunk: {chunks[0][:100]}...")
        logger.debug(f"First embedding shape: {len(embeddings[0])}")
        
        query_embedding = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
        logger.debug(f"Query embedding created. Shape: {len(query_embedding)}")
        
        relevant_chunks = find_most_relevant_chunks(query, query_embedding, chunks, embeddings)
        logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
        
        context = " ".join(relevant_chunks)
        full_context = f"Original text: {context}\n\nQuestion: {query}"
        logger.debug(f"Full context: {full_context[:500]}...")
        
        answer = generate_answer(query, full_context)
        logger.info("Successfully generated answer")
        logger.debug(f"Final generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"