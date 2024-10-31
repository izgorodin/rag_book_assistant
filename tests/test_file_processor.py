# tests/test_file_processor.py
import pytest
from src.services.file_processor import FileProcessor
import os
from src.utils.error_handler import FileProcessingError
from ebooklib import epub

@pytest.fixture
def file_processor():
    return FileProcessor()

@pytest.fixture
def sample_files(tmp_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'test_data')
    
    # Создаем тестовый EPUB файл
    epub_path = os.path.join(tmp_path, "test.epub")
    _create_test_epub(epub_path)
    
    return {
        "pdf": os.path.join(data_dir, "test.pdf"),
        "docx": os.path.join(data_dir, "test.docx"),
        "odt": os.path.join(data_dir, "test.odt"),
        "txt": os.path.join(data_dir, "test.txt"),
        "epub": epub_path
    }

def _create_test_epub(epub_path):
    """Создает тестовый EPUB файл"""
    book = epub.EpubBook()
    
    # Метаданные
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    
    # Создаем контент
    content = '''
        <h1>Test Book</h1>
        <p>This is a test paragraph with some content.</p>
        <p>It contains multiple paragraphs for testing.</p>
    '''
    
    # Создаем главу
    chapter = epub.EpubHtml(
        title='Test Chapter',
        file_name='chapter.xhtml',
        lang='en',
        content=content
    )
    book.add_item(chapter)
    
    # Добавляем навигацию
    book.toc = [(epub.Section('Test Chapter'), [chapter])]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # Добавляем стиль по умолчанию
    style = epub.EpubItem(
        uid="style_default",
        file_name="style/default.css",
        media_type="text/css",
        content=""
    )
    book.add_item(style)
    
    # Важно: правильный порядок в spine
    book.spine = ['nav', chapter]
    
    # Записываем EPUB с опцией игнорирования NCX
    epub.write_epub(epub_path, book, {'ignore_ncx': True})

def test_process_pdf(file_processor, sample_files):
    content = file_processor.process_file(sample_files["pdf"])
    assert isinstance(content, str)
    assert len(content) > 0

def test_process_docx(file_processor, sample_files):
    content = file_processor.process_file(sample_files["docx"])
    assert isinstance(content, str)
    assert len(content) > 0

def test_process_odt(file_processor, sample_files):
    content = file_processor.process_file(sample_files["odt"])
    assert isinstance(content, str)
    assert len(content) > 0

def test_process_txt(file_processor, sample_files):
    content = file_processor.process_file(sample_files["txt"])
    assert isinstance(content, str)
    assert len(content) > 0

def test_process_epub(file_processor, sample_files):
    """Тест обработки EPUB файлов"""
    content = file_processor.process_file(sample_files["epub"])
    assert isinstance(content, str)
    assert len(content) > 0
    assert "Test Book" in content
    assert "test paragraph" in content

def test_unsupported_file_format(file_processor, tmp_path):
    unsupported_file = tmp_path / "test.unsupported"
    unsupported_file.write_text("Unsupported content")
    
    with pytest.raises(FileProcessingError) as exc_info:
        file_processor.process_file(str(unsupported_file))
    
    error = exc_info.value
    assert "Unsupported file format" in str(error)
    assert ".unsupported" in error.details['extension']
    assert all(fmt in error.details['supported_formats'] 
              for fmt in ['.txt', '.pdf', '.docx', '.epub'])
