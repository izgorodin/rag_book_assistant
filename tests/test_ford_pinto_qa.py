import pytest
import spacy
from typing import List, Dict
from tests.ford_pinto_qa_data import qa_pairs
from src.rag import rag_query
from src.text_processing import split_into_chunks
from src.embedding import get_or_create_chunks_and_embeddings
from src.logger import setup_logger, setup_results_logger
from src.pinecone_manager import PineconeManager
from src.embedding import create_embeddings
from src.book_data_interface import BookDataInterface

nlp = spacy.load("en_core_web_md")  # Используйте 'md' или 'lg' вместо 'sm'

# Use the logger from logger.py
logger = setup_logger('test_rag_system.log')
results_logger = setup_results_logger()

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
    
    text_dict = {'text': text}
    
    chunks = split_into_chunks(text_dict)
    pinecone_manager = PineconeManager()
    embeddings = pinecone_manager.get_or_create_embeddings(chunks, lambda x: create_embeddings(x))
    return BookDataInterface(chunks, embeddings, {})

def get_answer_from_system(question: str, book_data: BookDataInterface) -> str:
    return rag_query(question, book_data)

@pytest.fixture(scope="module")
def system_setup():
    book_path = "tests/data/ford.txt"  # Убедитесь, что этот путь корректен
    return initialize_system(book_path)

@pytest.mark.parametrize("qa_pair", qa_pairs)
def test_qa_system(qa_pair, system_setup, mock_openai_service):
    book_data = system_setup
    question = qa_pair["question"]
    correct_answer = qa_pair["answer"]
    context = qa_pair.get("context", "")
    
    # Передаем mock_openai_service в rag_query
    system_answer = rag_query(question, book_data, mock_openai_service)
    
    is_correct = check_answer(system_answer, correct_answer, context)
    
    results_logger.info(f"\nQ: {question}")
    results_logger.info(f"Expected A: {correct_answer}")
    results_logger.info(f"System A: {system_answer}")
    results_logger.info(f"Is correct: {is_correct}")
    results_logger.info("-" * 50)
    
    assert is_correct, f"Incorrect answer for question: {question}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
