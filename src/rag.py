import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
import logging
from src.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def find_relevant_chunks(query: str, chunks: List[Dict[str, any]], top_k: int = 20) -> List[Dict[str, any]]:
    logger.info(f"Finding relevant chunks for query: {query}")
    
    preprocessed_query = preprocess_text(query)
    preprocessed_chunks = [preprocess_text(chunk['text']) for chunk in chunks]
    
    bm25 = BM25Okapi([chunk.split() for chunk in preprocessed_chunks])
    scores = bm25.get_scores(preprocessed_query.split())
    
    # Boost scores based on matched entities and key phrases
    query_entities = set(word_tokenize(query.lower()))
    for i, chunk in enumerate(chunks):
        chunk_entities = set(word.lower() for entity_list in chunk['entities'].values() for entity in entity_list for word in word_tokenize(entity))
        chunk_phrases = set(word.lower() for phrase in chunk['key_phrases'] for word in word_tokenize(phrase))
        
        entity_match = len(query_entities.intersection(chunk_entities))
        phrase_match = len(query_entities.intersection(chunk_phrases))
        
        scores[i] += 0.1 * entity_match + 0.05 * phrase_match
        
        if chunk['dates']:
            scores[i] += 0.1
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    relevant_chunks = [{'chunk': chunks[i], 'score': scores[i]} for i in top_indices]
    
    logger.info(f"Found {len(relevant_chunks)} relevant chunks")
    return relevant_chunks

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in extracting precise information from texts. Focus on providing accurate information. If the exact information is not available, explain what is known and what is missing."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nProvide a concise answer based on the context. If specific information is not available, briefly explain what is known and what is missing."}
    ]
    
    logger.info(f"Generating answer for query: {query}")
    
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
        
        hybrid_search = HybridSearch(chunks, embeddings)
        relevant_chunks = hybrid_search.search(query, top_k=5)
        
        context = "\n\n".join(f"Chunk {i+1} (score: {chunk['score']:.2f}): {chunk['chunk']}" for i, chunk in enumerate(relevant_chunks))
        
        full_context = f"Original text chunks:\n\n{context}\n\nQuestion: {query}"
        
        answer = generate_answer(query, full_context)
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        if "OpenAI API" in str(e):
            return "I'm sorry, but the service is currently unavailable. Please try again later."
        return f"Sorry, I encountered an error while processing your query: {str(e)}"
