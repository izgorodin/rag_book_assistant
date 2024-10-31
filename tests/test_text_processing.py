import pytest
import os
from ebooklib import epub
from src.services.text_processor import load_and_preprocess_text, split_into_chunks
from src.services.epub_processor import EPUBProcessor

def test_load_and_preprocess_text():
    test_file_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test_book.txt')
    with open(test_file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Test with raw text
    processed_text = load_and_preprocess_text(text_content)
    _verify_processed_text(processed_text)
    
    # Test with already processed text
    reprocessed_text = load_and_preprocess_text(processed_text)
    _verify_processed_text(reprocessed_text)

def _verify_processed_text(processed_text):
    assert isinstance(processed_text, dict), "Processed text should be a dictionary"
    assert 'text' in processed_text, "Processed text should contain 'text' key"
    assert isinstance(processed_text['text'], str), "The 'text' value should be a string"
    assert len(processed_text['text']) > 0, "Processed text should not be empty"
    assert 'chunks' in processed_text, "Processed text should contain 'chunks' key"
    assert isinstance(processed_text['chunks'], list), "Chunks should be a list"

@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 50),
    (500, 100)
])
def test_split_into_chunks(chunk_size, overlap):
    text = ' '.join(['word'] * 1000)  # Create a sample text
    chunks = split_into_chunks(text, chunk_size, overlap)

    assert isinstance(chunks, list), "Function should return a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    
    # Check that all chunks do not exceed the maximum size
    assert all(len(chunk.split()) <= chunk_size for chunk in chunks), "Chunks should not exceed max size"
    
    # Check that most chunks (except the last one) are close to the maximum size
    assert all(len(chunk.split()) >= chunk_size * 0.9 for chunk in chunks[:-1]), "Most chunks should be close to max size"
    
    # Check the overlap between chunks
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            assert chunks[i].split()[-overlap:] == chunks[i+1].split()[:overlap], f"Chunks {i} and {i+1} should overlap correctly"

def test_split_into_chunks_with_dict_input():
    text_dict = {'text': ' '.join(['word'] * 1000)}
    chunks = split_into_chunks(text_dict, chunk_size=100, overlap=20)
    assert isinstance(chunks, list), "Function should return a list"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"

def _verify_book_data(book_data):
    # Test basic properties
    assert len(book_data.get_chunks()) > 0
    assert len(book_data.get_embeddings()) == len(book_data.get_chunks())
    assert all(len(emb) == 1536 for emb in book_data.get_embeddings())
    # Здесь можно добавить более конкретные проверки, если содержание test_book.txt известно
    # Test new features
    assert isinstance(book_data.get_dates(), list)
    assert isinstance(book_data.get_entities(), list)
    assert isinstance(book_data.get_key_phrases(), list)

def test_epub_processing(tmp_path):
    """Тест обработки EPUB файлов"""
    # Создаем тестовый EPUB
    epub_path = _create_test_epub(tmp_path)
    
    # Обрабатываем EPUB
    processor = EPUBProcessor()
    epub_text = processor.process_epub(epub_path)
    
    # Проверяем, что текст извлечен
    assert isinstance(epub_text, str)
    assert len(epub_text) > 0
    
    # Проверяем предобработку текста
    processed_text = load_and_preprocess_text(epub_text)
    _verify_processed_text(processed_text)
    
    # Проверяем разбиение на чанки
    chunks = split_into_chunks(processed_text, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)

def _create_test_epub(tmp_path):
    """Создает тестовый EPUB файл"""
    book = epub.EpubBook()
    
    # Метаданные
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    
    # Создаем контент
    content = '''
        <h1>Test Book</h1>
        <p>This is a test paragraph with some important information.</p>
        <p>It contains multiple paragraphs to test text extraction.</p>
        <h2>Chapter 1</h2>
        <p>The first chapter contains technical details about the system.</p>
        <p>It includes various terms and concepts that need to be processed.</p>
        <h2>Chapter 2</h2>
        <p>The second chapter discusses implementation details.</p>
        <p>It provides examples and use cases for better understanding.</p>
    '''
    
    # Создаем главу
    chapter = epub.EpubHtml(
        title='Test Chapter',
        file_name='chapter.xhtml',
        lang='en',
        content=content
    )
    book.add_item(chapter)
    
    # Добавляем в spine
    book.spine = [chapter]
    
    # Создаем путь для файла
    epub_path = os.path.join(tmp_path, 'test.epub')
    
    # Записываем EPUB
    epub.write_epub(epub_path, book, {})
    
    return epub_path

def test_epub_processing_with_multiple_chapters(tmp_path):
    """Тест обработки EPUB с несколькими главами"""
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    
    chapters = []
    for i in range(3):
        content = f'''
            <h1>Chapter {i+1}</h1>
            <p>This is the content of chapter {i+1}.</p>
            <p>It contains multiple paragraphs for testing.</p>
            <p>Each chapter has unique content for verification.</p>
        '''
        chapter = epub.EpubHtml(
            title=f'Chapter {i+1}',
            file_name=f'chapter_{i+1}.xhtml',
            lang='en',
            content=content
        )
        book.add_item(chapter)
        chapters.append(chapter)
    
    book.spine = chapters
    epub_path = os.path.join(tmp_path, 'multi_chapter.epub')
    epub.write_epub(epub_path, book, {})
    
    processor = EPUBProcessor()
    text = processor.process_epub(epub_path)
    
    # Проверяем наличие контента из всех глав
    for i in range(3):
        assert f'Chapter {i+1}' in text
        assert f'content of chapter {i+1}' in text
    
    # Проверяем предобработку
    processed_text = load_and_preprocess_text(text)
    _verify_processed_text(processed_text)

def test_epub_processing_with_empty_content(tmp_path):
    """Тест обработки EPUB с пустым содержимым"""
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Empty Book')
    book.set_language('en')
    
    chapter = epub.EpubHtml(
        title='Empty Chapter',
        file_name='chapter.xhtml',
        lang='en',
        content='<html><body></body></html>'
    )
    book.add_item(chapter)
    book.spine = [chapter]
    
    epub_path = os.path.join(tmp_path, 'empty.epub')
    epub.write_epub(epub_path, book, {})
    
    processor = EPUBProcessor()
    text = processor.process_epub(epub_path)
    
    # Проверяем, что получаем пустой текст
    assert text.strip() == ""
    
    # Проверяем предобработку пустого текста
    processed_text = load_and_preprocess_text(text)
    assert processed_text['text'].strip() == ""
    assert len(processed_text['chunks']) == 0
