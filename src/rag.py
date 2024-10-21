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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.logger_config import setup_logger

logger = setup_logger()

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
    logger.debug(f"Preprocessed query: {preprocessed_query}")
    
    preprocessed_chunks = [preprocess_text(chunk['text']) for chunk in chunks]
    
    # BM25 scoring
    bm25 = BM25Okapi([chunk.split() for chunk in preprocessed_chunks])
    bm25_scores = bm25.get_scores(preprocessed_query.split())
    logger.debug(f"BM25 scores (top 3): {sorted(bm25_scores, reverse=True)[:3]}")
    
    # TF-IDF scoring
    tfidf = TfidfVectorizer().fit(preprocessed_chunks)
    chunk_vectors = tfidf.transform(preprocessed_chunks)
    query_vector = tfidf.transform([preprocessed_query])
    tfidf_scores = cosine_similarity(query_vector, chunk_vectors)[0]
    logger.debug(f"TF-IDF scores (top 3): {sorted(tfidf_scores, reverse=True)[:3]}")
    
    # Combine scores
    combined_scores = 0.7 * np.array(bm25_scores) + 0.3 * tfidf_scores
    logger.debug(f"Combined scores (top 3): {sorted(combined_scores, reverse=True)[:3]}")
    
    # Boost scores based on matched entities and key phrases
    query_entities = set(word_tokenize(query.lower()))
    for i, chunk in enumerate(chunks):
        chunk_entities = set(word.lower() for entity_list in chunk['entities'].values() for entity in entity_list for word in word_tokenize(entity))
        chunk_phrases = set(word.lower() for phrase in chunk['key_phrases'] for word in word_tokenize(phrase))
        
        entity_match = len(query_entities.intersection(chunk_entities))
        phrase_match = len(query_entities.intersection(chunk_phrases))
        
        combined_scores[i] += 0.1 * entity_match + 0.05 * phrase_match
        
        if chunk['dates']:
            combined_scores[i] += 0.1
    
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    relevant_chunks = [{'chunk': chunks[i], 'score': combined_scores[i]} for i in top_indices]
    
    logger.info(f"Found {len(relevant_chunks)} relevant chunks")
    for i, chunk in enumerate(relevant_chunks[:3]):
        logger.debug(f"Top {i+1} chunk score: {chunk['score']:.4f}")
        logger.debug(f"Top {i+1} chunk text: {chunk['chunk']['text'][:100]}...")
    
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
        relevant_chunks = hybrid_search.search(query, top_k=10)
        logger.info(f"Number of relevant chunks found: {len(relevant_chunks)}")
        
        for i, chunk in enumerate(relevant_chunks):
            logger.debug(f"Chunk {i+1} (score: {chunk['score']:.2f}):")
            logger.debug(f"Content: {chunk['chunk'][:200]}...")
        
        context = "\n\n".join(f"Chunk {i+1} (score: {chunk['score']:.2f}): {chunk['chunk']}" for i, chunk in enumerate(relevant_chunks))
        logger.debug(f"Full context:\n{context}")
        
        full_context = f"Original text chunks:\n\n{context}\n\nQuestion: {query}"
        
        answer = generate_answer(query, full_context)
        logger.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"
