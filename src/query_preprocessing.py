import nltk
from nltk.corpus import wordnet
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from src.config import OPENAI_API_KEY, GPT_MODEL
import json

# Убедимся, что необходимые ресурсы NLTK загружены
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

client = OpenAI(api_key=OPENAI_API_KEY)

def expand_query_with_synonyms(query: str) -> str:
    """
    Расширяет запрос, добавляя синонимы к ключевым словам.
    """
    tokens = nltk.word_tokenize(query)
    pos_tags = nltk.pos_tag(tokens)
    
    expanded_query = []
    for word, pos in pos_tags:
        expanded_query.append(word)
        # Ищем синонимы только для существительных, глаголов и прилагательных
        if pos.startswith('N') or pos.startswith('V') or pos.startswith('J'):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.add(lemma.name())
            # Добавляем до 3 синонимов
            expanded_query.extend(list(synonyms)[:3])
    
    return ' '.join(expanded_query)

def analyze_user_intent(query: str) -> Tuple[str, str]:
    """
    Простой анализ намерений пользователя на основе ключевых слов.
    """
    query_lower = query.lower()
    if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
        intent = 'question'
    elif any(word in query_lower for word in ['find', 'search', 'look for']):
        intent = 'search'
    elif any(word in query_lower for word in ['summarize', 'summary']):
        intent = 'summarize'
    else:
        intent = 'general'
    
    return intent, query

def optimize_query(query: str, text_metadata: dict) -> str:
    """
    Оптимизирует запрос с использованием GPT и метаданных текста.
    """
    prompt = f"""
    Given the following user query and text metadata, create an optimized search query for our hybrid search system.

    User Query: {query}

    Text Metadata:
    {json.dumps(text_metadata, indent=2)}

    Our hybrid search system works as follows:
    1. It combines BM25 algorithm for keyword matching with embedding-based semantic search.
    2. The system expands queries with synonyms and related terms.
    3. It uses a weighted combination of BM25 scores and cosine similarity of embeddings.
    4. The search considers entities, key phrases, and dates from the text metadata.
    5. Query terms are weighted based on their part of speech (nouns, verbs, adjectives have higher weights).

    Guidelines for creating an optimized query:
    1. Identify and include key concepts from the original query.
    2. Use relevant entities, key phrases, and dates from the metadata if applicable.
    3. Expand on important terms using synonyms or related concepts.
    4. Consider the intent of the query (e.g., question, search, summarize).
    5. Include terms that would work well with both keyword matching (BM25) and semantic search.
    6. Prioritize nouns, verbs, and adjectives as they get higher weights in our system.
    7. Ensure the optimized query is concise but comprehensive.
    8. Format the optimized query as a single string, ready for search.

    Optimized Query:
    """
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in optimizing search queries for hybrid search systems."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        optimized_query = response.choices[0].message.content.strip()
        return optimized_query
    except Exception as e:
        print(f"Error in optimize_query: {str(e)}")
        return query  # В случае ошибки возвращаем исходный запрос

def preprocess_query(query: str, text_metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Предобрабатывает запрос: оптимизирует его, расширяет синонимами и определяет намерение пользователя.
    """
    intent, original_query = analyze_user_intent(query)
    optimized_query = optimize_query(original_query, text_metadata)
    expanded_query = expand_query_with_synonyms(optimized_query)
    return original_query, expanded_query, intent
