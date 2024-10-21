import pytest
import spacy
from typing import List, Dict
from tests.ford_pinto_qa_data import qa_pairs
from src.rag import rag_query
from src.text_processing import load_and_preprocess_text, split_into_chunks
from src.embedding import get_or_create_chunks_and_embeddings

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> Dict[str, str]:
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def check_answer(system_answer: str, correct_answer: str, context: str) -> bool:
    system_entities = extract_entities(system_answer)
    correct_entities = extract_entities(correct_answer)
    
    for entity_type, entity_value in correct_entities.items():
        if entity_type not in system_entities or system_entities[entity_type] != entity_value:
            return False
    
    context_doc = nlp(context)
    answer_doc = nlp(system_answer)
    
    for sent in context_doc.sents:
        if correct_answer.lower() in sent.text.lower() and any(token.text.lower() in sent.text.lower() for token in answer_doc):
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
    context = qa_pair.get("context", "")  # Используем .get() с значением по умолчанию
    
    system_answer = get_answer_from_system(question, chunks, embeddings)
    is_correct = check_answer(system_answer, correct_answer, context)
    similarity_score = semantic_similarity(system_answer, correct_answer)
    
    print(f"Q: {question}")
    print(f"System A: {system_answer}")
    print(f"Correct A: {correct_answer}")
    print(f"Is correct: {is_correct}")
    print(f"Similarity score: {similarity_score:.2f}\n")
    
    assert is_correct, f"Incorrect answer for question: {question}"
    assert similarity_score > 0.5, f"Low similarity score for question: {question}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
