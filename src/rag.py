import re
from typing import List, Tuple
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

def extract_dates(text: str) -> List[str]:
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
    return re.findall(date_pattern, text)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
    logger.info(f"Finding relevant chunks for query: {query}")
    
    preprocessed_query = preprocess_text(query)
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    
    bm25 = BM25Okapi([chunk.split() for chunk in preprocessed_chunks])
    scores = bm25.get_scores(preprocessed_query.split())
    
    date_boost = 0.5
    for i, chunk in enumerate(chunks):
        if extract_dates(chunk):
            scores[i] += date_boost
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    relevant_chunks = [(chunks[i], scores[i]) for i in top_indices]
    
    logger.info(f"Found {len(relevant_chunks)} relevant chunks")
    return relevant_chunks

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in extracting precise information from legal and historical texts. Focus on providing accurate dates and events. If the exact date is not available, explain what is known and what is missing. Always mention the most relevant dates found in the context, even if they don't directly answer the question."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nProvide a concise answer based on the context. If the exact date is not available, briefly explain what is known and what is missing. Always mention the most relevant dates found in the context, even if they don't directly answer the question."}
    ]
    
    logger.info(f"Generating answer for query: {query}")
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

def postprocess_answer(answer: str, chunks: List[str]) -> str:
    if not any(extract_dates(answer)):
        all_dates = []
        for chunk in chunks:
            all_dates.extend(extract_dates(chunk))
        if all_dates:
            return f"{answer}\n\nWhile I couldn't find the specific date you asked for, here are some relevant dates mentioned in the text: {', '.join(all_dates[:5])}"
    return answer

def rag_query(query: str, chunks: List[str]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        
        relevant_chunks = find_relevant_chunks(query, chunks)
        context = "\n\n".join(f"Chunk {i+1} (score: {score:.2f}): {chunk}" for i, (chunk, score) in enumerate(relevant_chunks))
        full_context = f"Original text chunks:\n\n{context}\n\nQuestion: {query}"
        
        answer = generate_answer(query, full_context)
        processed_answer = postprocess_answer(answer, chunks)
        logger.info("Successfully generated answer")
        return processed_answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"