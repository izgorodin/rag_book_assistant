import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 10) -> List[str]:
    logger.debug(f"Finding relevant chunks for query: {query}")
    
    # Preprocess query and chunks
    preprocessed_query = preprocess_text(query)
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit(preprocessed_chunks + [preprocessed_query])
    chunk_vectors = vectorizer.transform(preprocessed_chunks)
    query_vector = vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    # Apply position-based weights
    position_weights = np.linspace(1, 0.5, len(chunks))
    weighted_similarities = similarities * position_weights
    
    # Get top-k chunks
    top_indices = weighted_similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
    for i, chunk in enumerate(relevant_chunks):
        logger.debug(f"Relevant chunk {i} (similarity: {similarities[top_indices[i]]:.4f}): {chunk[:100]}...")
    
    return relevant_chunks

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the exact answer is not in the context, provide the most relevant information available and explain any uncertainties."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nIf you can't find the exact answer, provide the most relevant information from the context and explain what's missing."}
    ]
    
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

def rag_query(query: str, chunks: List[str]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        logger.debug(f"Total chunks: {len(chunks)}")
        
        relevant_chunks = find_relevant_chunks(query, chunks)
        logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
        
        context = "\n\n".join(f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks))
        full_context = f"Original text chunks:\n\n{context}\n\nQuestion: {query}"
        logger.debug(f"Full context: {full_context[:500]}...")
        
        answer = generate_answer(query, full_context)
        logger.info("Successfully generated answer")
        logger.debug(f"Final generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"