#import pytest
#import os
#from unittest.mock import MagicMock
#from src.embedding import EmbeddingService
#from src.rag import rag_query
#from src.book_data_interface import BookDataInterface
#from src.book_data_factory import BookDataFactory
#from src.vector_store_service import VectorStoreService
#from tests.utils.mock_factory import MockFactory

    #TEST_FILES_DIR = os.path.dirname(os.path.abspath(__file__))

    #@pytest.fixture
    #def mock_services():
    #embedding_service = MagicMock(spec=EmbeddingService)
    #embedding_service.create_embeddings.return_value = [[0.1] * 1536]
    
    #mock_pinecone = MockFactory.create_pinecone_client()
    #vector_store_service = VectorStoreService(mock_pinecone)
    
    #return embedding_service, vector_store_service

    #@pytest.fixture
    #def book_factory(mock_services):
    #embedding_service, vector_store_service = mock_services
    #return BookDataFactory(
    #    embedding_service=embedding_service,
    #    vector_store_service=vector_store_service
    #)

    #def test_book_processing(book_factory, tmp_path):
    # Create test book
    #test_book = tmp_path / "test_book.txt"
    #test_book.write_text("This is a test book about AI and machine learning. Published in 2023.")
    
    #with open(test_book, 'r') as f:
    #    book_data = book_factory.create_from_text(f.read())
    
    # Test basic properties
    #assert len(book_data.get_chunks()) > 0
    #assert len(book_data.get_embeddings()) == len(book_data.get_chunks())
    #assert all(len(emb) == 1536 for emb in book_data.get_embeddings())
    
    # Test new features
    #assert isinstance(book_data.get_dates(), list)
    #assert isinstance(book_data.get_entities(), list)
    #assert isinstance(book_data.get_key_phrases(), list)
    #assert "2023" in book_data.get_dates()
    #assert any("AI" in str(entity) for entity in book_data.get_entities())

    #def test_book_data_persistence(book_factory, tmp_path):
    # Create and save book data
    #test_text = "Test book content"
    #book_data = book_factory.create_from_text(test_text)
    
    #save_path = tmp_path / "test_book.pkl"
    #book_data.save(str(save_path))
    
    # Load and verify
    #loaded_data = BookDataInterface.from_file(str(save_path))
    #assert loaded_data.get_chunks() == book_data.get_chunks()
    #assert loaded_data.get_embeddings() == book_data.get_embeddings()
    #assert loaded_data.get_dates() == book_data.get_dates()
    #assert loaded_data.get_entities() == book_data.get_entities()
    #assert loaded_data.get_key_phrases() == book_data.get_key_phrases()

    #def test_rag_query_integration(book_factory):
    #"""Тест интеграции RAG запроса."""
    # Подготовка данных
    #test_text = "AI and machine learning are transforming industries."
    #book_data = book_factory.create_from_text(test_text)
    
    # Создаем сервисы
    #embedding_service = MockFactory.create_embedding_service()
    #openai_service = MockFactory.create_openai_client()
    
    # Выполняем запрос через функцию из модуля rag
    #from src.rag import rag_query
    #response = rag_query(
    #    query="What is transforming industries?",
    #    book_data=book_data,
    #    openai_service=openai_service,
    #    embedding_service=embedding_service
    #)
    
    # Проверки
    #assert isinstance(response, str)
    #assert len(response) > 0

    #def test_error_handling(book_factory):
    #with pytest.raises(ValueError):
    #    book_factory.create_from_text("")  # Empty text should raise error

    #def test_book_processing_raw_text(book_factory, tmp_path):
    # Test with raw text input
    #test_content = "This is a test book about AI and machine learning. Published in 2023."
    #book_data = book_factory.create_from_text(test_content)
    
    #_verify_book_data(book_data)

    #def test_book_processing_preprocessed_data(book_factory):
    # Test with preprocessed data input
    #preprocessed_data = {
    #    'text': "This is a test book about AI and machine learning. Published in 2023.",
    #    'chunks': ["This is a test book about AI", "machine learning. Published in 2023."]
    #}
    #book_data = book_factory.create_from_text(preprocessed_data)
    
    #_verify_book_data(book_data)

    #def _verify_book_data(book_data):
    # Test basic properties
    #assert len(book_data.get_chunks()) > 0
    #assert len(book_data.get_embeddings()) == len(book_data.get_chunks())
    #assert all(len(emb) == 1536 for emb in book_data.get_embeddings())
    
    # Test new features
    #assert isinstance(book_data.get_dates(), list)
    #assert isinstance(book_data.get_entities(), list)
    #assert isinstance(book_data.get_key_phrases(), list)

    #def test_error_handling_invalid_input(book_factory):
    #with pytest.raises(ValueError) as exc_info:
    #    book_factory.create_from_text(['invalid', 'input'])
    #assert "Expected string or preprocessed data dict" in str(exc_info.value)
