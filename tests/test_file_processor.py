# tests/test_file_processor.py
import pytest
from src.file_processor import FileProcessor
import os

@pytest.fixture
def file_processor():
    return FileProcessor()

@pytest.fixture
def sample_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    return {
        "pdf": os.path.join(data_dir, "test.pdf"),
        "docx": os.path.join(data_dir, "test.docx"),
        "odt": os.path.join(data_dir, "test.odt"),
        "txt": os.path.join(data_dir, "test.txt")
    }

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

def test_unsupported_file_format(file_processor, tmp_path):
    unsupported_file = tmp_path / "test.unsupported"
    unsupported_file.write_text("Unsupported content")
    with pytest.raises(ValueError, match="Unsupported file format"):
        file_processor.process_file(str(unsupported_file))
