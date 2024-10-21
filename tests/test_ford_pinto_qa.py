import pytest
import spacy
from typing import List, Dict
from tests.ford_pinto_qa_data import qa_pairs
from src.rag import rag_query
from src.text_processing import split_into_chunks
from src.embedding import get_or_create_chunks_and_embeddings
from src.logger_config import setup_logger
import re

nlp = spacy.load("en_core_web_md")  # Используйте 'md' или 'lg' вместо 'sm'

logger = setup_logger('test_rag_system.log')

def extract_entities(text: str) -> Dict[str, str]:
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def check_answer(system_answer: str, correct_answer: str, context: str) -> bool:
    system_answer = system_answer.lower()
    correct_answer = correct_answer.lower()
    
    # Проверка на точное совпадение или наличие правильного ответа в системном ответе
    if correct_answer in system_answer:
        return True
    
    # Проверка на наличие ключевых слов
    keywords = set(correct_answer.split()) - set(['a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    if all(keyword in system_answer for keyword in keywords):
        return True
    
    # Проверка на наличие имени в ответе
    system_entities = extract_entities(system_answer)
    correct_entities = extract_entities(correct_answer)
    if any(entity in system_entities.values() for entity in correct_entities.values()):
        return True
    
    # Проверка на семантическое сходство
    if semantic_similarity(system_answer, correct_answer) > 0.7:
        return True
    
    return False

def semantic_similarity(text1: str, text2: str) -> float:
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def initialize_system(book_path):
    with open(book_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Создаем словарь с ключом 'text'
    text_dict = {'text': text}
    
    chunks = split_into_chunks(text_dict)
    chunks, embeddings = get_or_create_chunks_and_embeddings(chunks, 'embeddings_cache.pkl')
    return chunks, embeddings

def get_answer_from_system(question: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    return rag_query(question, chunks, embeddings)

@pytest.fixture(scope="module")
def system_setup():
    book_path = "tests/ford.txt"  # Убедитесь, что этот путь корректен
    return initialize_system(book_path)

@pytest.mark.parametrize("qa_pair", qa_pairs)
def test_qa_system(qa_pair, system_setup):
    chunks, embeddings = system_setup
    question = qa_pair["question"]
    correct_answer = qa_pair["answer"]
    context = qa_pair.get("context", "")
    
    logger.info(f"Testing question: {question}")
    logger.debug(f"Correct answer: {correct_answer}")
    
    system_answer = get_answer_from_system(question, chunks, embeddings)
    is_correct = check_answer(system_answer, correct_answer, context)
    
    logger.info(f"System answer: {system_answer}")
    logger.info(f"Is correct: {is_correct}")
    
    print(f"\nQ: {question}")
    print(f"System A: {system_answer}")
    print(f"Correct A: {correct_answer}")
    print(f"Is correct: {is_correct}")
    
    assert is_correct, f"Incorrect answer for question: {question}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
