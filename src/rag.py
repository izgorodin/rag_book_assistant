"""
RAG (Retrieval-Augmented Generation) Module

This module implements the final stage of the RAG system for question answering based on a given text corpus.
It utilizes the results from text processing and hybrid search to generate accurate and contextually relevant
answers to user queries.

Key components of the overall RAG system:
1. Text processing (src/text_processing.py):
   - Processing large files
   - Splitting text into chunks
   - Extracting dates, named entities, and key phrases
2. Hybrid search (src/hybrid_search.py):
   - Combining semantic (embedding-based) and lexical (BM25) search
   - Query expansion using synonyms
   - Weighting query tokens based on their parts of speech
3. RAG query processing (this module):
   - Using HybridSearch to find relevant chunks
   - Constructing context from relevant chunks and metadata
   - Generating answers using OpenAI's GPT model

The main function, rag_query, orchestrates the process from relevant chunk retrieval to answer generation.
"""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
import logging
from src.embedding import create_embeddings
from src.hybrid_search import HybridSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.logger_config import setup_logger
from src.book_data_interface import BookDataInterface
from src.pinecone_manager import PineconeManager

logger = setup_logger()

client = OpenAI(api_key=OPENAI_API_KEY)

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by converting to lowercase, removing punctuation,
    tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer to the given query based on the provided context using OpenAI's GPT model.

    Args:
        query (str): The user's question.
        context (str): The context information retrieved from the document.

    Returns:
        str: The generated answer or an error message if generation fails.
    """
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

def rag_query(query: str, book_data: BookDataInterface) -> str:
    """
    Process a RAG query by finding relevant chunks, constructing context, and generating an answer.

    Args:
        query (str): The user's question.
        book_data (BookDataInterface): The book data interface object containing chunks, embeddings, and processed text.

    Returns:
        str: The generated answer or an error message if processing fails.
    """
    try:
        logger.info(f"Processing RAG query: {query}")
        
        pinecone_manager = PineconeManager()
        query_embedding = create_embeddings([query])[0]
        relevant_chunks = pinecone_manager.search_similar(query_embedding, top_k=10)
        logger.info(f"Number of relevant chunks found: {len(relevant_chunks)}")
        
        # Log the top 3 most relevant chunks
        for i, chunk in enumerate(relevant_chunks[:3]):
            logger.debug(f"Chunk {i+1} (score: {chunk['score']:.2f}):")
            logger.debug(f"Content: {chunk['chunk'][:200]}...")
        
        # Construct context from relevant chunks
        context = "\n\n".join(f"Chunk {i+1} (score: {chunk['score']:.2f}): {chunk['chunk']}" for i, chunk in enumerate(relevant_chunks))
        
        # Add metadata to context
        context += f"\n\nDates mentioned: {', '.join(book_data.processed_text.get('dates', []))}"
        context += f"\n\nKey phrases: {', '.join(book_data.processed_text.get('key_phrases', []))}"
        context += "\n\nNamed entities:"
        for entity_type, entities in book_data.processed_text.get('entities', {}).items():
            if entities:
                context += f"\n- {entity_type}: {', '.join(entities[:5])}"
        
        full_context = f"Original text chunks and metadata:\n\n{context}\n\nQuestion: {query}"
        
        # Generate answer using the constructed context
        answer = generate_answer(query, full_context)
        logger.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"
