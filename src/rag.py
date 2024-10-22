import re
from typing import Any, List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL, MAX_TOKENS
from src.hybrid_search import HybridSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.logger_config import setup_logger
from src.query_preprocessing import preprocess_query
from src.metadata_extractor import extract_metadata
import tiktoken
from collections import defaultdict, Counter
import json

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

def extract_concise_metadata(text: str) -> Dict[str, Any]:
    try:
        tokens = word_tokenize(text)
        
        # Простое извлечение именованных сущностей (заглавные слова)
        entities = [word for word in tokens if word[0].isupper()]
        
        # Извлечение ключевых фраз (биграммы и триграммы)
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        key_phrases = bigrams + trigrams
        
        # Получаем наиболее частые слова (исключая стоп-слова)
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
        word_freq = Counter(words)
        top_words = [word for word, _ in word_freq.most_common(10)]
        
        return {
            "entities": entities[:10],  # Ограничиваем до 10 сущностей
            "key_phrases": key_phrases[:20],  # Ограничиваем до 20 ключевых фраз
            "top_words": top_words,
            "document_length": len(tokens)
        }
    except Exception as e:
        logger.error(f"Error in extract_concise_metadata: {str(e)}")
        return {
            "entities": [],
            "key_phrases": [],
            "top_words": [],
            "document_length": len(text.split())
        }

def rag_query(query: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    try:
        logger.info(f"Processing RAG query: {query}")
        
        # Извлекаем метаданные из чанков
        text_metadata = [extract_concise_metadata(chunk) for chunk in chunks]
        
        # Объединяем метаданные всех чанков
        combined_metadata = {
            "entities": set(),
            "key_phrases": set(),
            "top_words": set(),
            "document_length": sum(metadata["document_length"] for metadata in text_metadata)
        }
        for metadata in text_metadata:
            combined_metadata["entities"].update(metadata["entities"])
            combined_metadata["key_phrases"].update(metadata["key_phrases"])
            combined_metadata["top_words"].update(metadata["top_words"])
        
        # Преобразуем множества обратно в списки и ограничиваем количество элементов
        combined_metadata["entities"] = list(combined_metadata["entities"])[:20]
        combined_metadata["key_phrases"] = list(combined_metadata["key_phrases"])[:30]
        combined_metadata["top_words"] = list(combined_metadata["top_words"])[:20]
        
        logger.info(f"Combined metadata: {json.dumps(combined_metadata, ensure_ascii=False, indent=2)}")
        
        # Предобработка запроса с использованием объединенных метаданных
        original_query, expanded_query, intent = preprocess_query(query, combined_metadata)
        logger.info(f"Original query: {original_query}")
        logger.info(f"Expanded query: {expanded_query}")
        logger.info(f"Detected intent: {intent}")
        
        # Используем расширенный запрос для поиска релевантных чанков
        hybrid_search = HybridSearch(chunks, embeddings)
        relevant_chunks = hybrid_search.search(expanded_query, top_k=5)
        logger.info(f"Number of relevant chunks found: {len(relevant_chunks)}")
        
        # Формируем контекст с ограничением токенов
        context = format_context(relevant_chunks, original_query, combined_metadata)
        logger.info(f"Formatted context: {context[:500]}...")  # Логируем первые 500 символов контекста
        
        # Формируем промпт в зависимости от намерения пользователя
        prompt = format_prompt(intent, context, original_query)
        logger.info(f"Formatted prompt: {prompt}")
        
        # Генерируем ответ с помощью GPT
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in extracting precise information from texts. Focus on providing accurate and concise answers. If the exact information is not available, explain what is known and what is missing."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info("Sending request to OpenAI API with messages:")
        logger.info(json.dumps(messages, ensure_ascii=False, indent=2))
        
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error in RAG query process: {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"

def format_context(relevant_chunks: List[Dict], query: str, metadata: Dict) -> str:
    context_parts = []
    total_tokens = 0
    max_tokens = 3000  # Adjust based on your GPT model's limit
    
    # Add relevant metadata
    relevant_entities = [ent for ent in metadata['entities'] if ent.lower() in query.lower()]
    relevant_phrases = [phrase for phrase in metadata['key_phrases'] if phrase.lower() in query.lower()]
    
    if relevant_entities or relevant_phrases:
        context_parts.append("Relevant metadata:")
        if relevant_entities:
            context_parts.append(f"Entities: {', '.join(relevant_entities)}")
        if relevant_phrases:
            context_parts.append(f"Key phrases: {', '.join(relevant_phrases)}")
        context_parts.append("")
    
    # Add relevant chunks
    for i, chunk in enumerate(relevant_chunks):
        chunk_text = f"Chunk {i+1} (relevance: {chunk['score']:.2f}):\n{chunk['chunk']}\n"
        chunk_tokens = len(tiktoken.encoding_for_model(GPT_MODEL).encode(chunk_text))
        if total_tokens + chunk_tokens > max_tokens:
            break
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    return "\n".join(context_parts)

def format_prompt(intent: str, context: str, query: str) -> str:
    if intent == 'question':
        return f"Based on the following context, please answer the question accurately and concisely. If the information is not available in the context, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    elif intent == 'search':
        return f"Based on the following context, please find and summarize the most relevant information for the query. If specific information is not available, mention what is known and what is missing.\n\nContext:\n{context}\n\nQuery: {query}\n\nRelevant information:"
    elif intent == 'summarize':
        return f"Please summarize the key points from the following context, focusing on the most important and relevant information.\n\nContext:\n{context}\n\nSummary:"
    else:
        return f"Based on the following context, please provide a relevant and informative response to the user's input. If specific information is not available, explain what is known and what is missing.\n\nContext:\n{context}\n\nUser input: {query}\n\nResponse:"

