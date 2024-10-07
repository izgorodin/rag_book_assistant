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

def generate_answer(query: str, context: str, entities: Dict[str, List[str]], key_phrases: List[str]) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in extracting precise information from legal and historical texts. Focus on providing accurate dates, events, and named entities. If the exact information is not available, explain what is known and what is missing."},
        {"role": "user", "content": f"Context: {context}\n\nRelevant entities: {entities}\n\nKey phrases: {key_phrases}\n\nQuestion: {query}\n\nProvide a concise answer based on the context. If specific information is not available, briefly explain what is known and what is missing. Try to incorporate relevant named entities and key phrases in your answer."}
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

def rag_query(query: str, chunks: List[Dict[str, any]]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        
        relevant_chunks = find_relevant_chunks(query, chunks)
        context = "\n\n".join(f"Chunk {i+1} (score: {chunk['score']:.2f}): {chunk['chunk']['text']}" for i, chunk in enumerate(relevant_chunks))
        
        all_entities = {}
        all_key_phrases = set()
        for chunk in relevant_chunks:
            for entity_type, entities in chunk['chunk']['entities'].items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = set()
                all_entities[entity_type].update(entities)
            all_key_phrases.update(chunk['chunk']['key_phrases'])
        
        all_entities = {k: list(v) for k, v in all_entities.items()}
        all_key_phrases = list(all_key_phrases)
        
        full_context = f"Original text chunks:\n\n{context}\n\nQuestion: {query}"
        
        answer = generate_answer(query, full_context, all_entities, all_key_phrases)
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"