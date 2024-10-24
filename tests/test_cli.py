import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import os
from src.cli import main, run_cli, load_and_process_book, answer_question
from src.book_data_interface import BookDataInterface

@pytest.fixture
def mock_book_data():
    return MagicMock(spec=BookDataInterface)

def test_main_cli_mode():
    with patch('sys.argv', ['cli.py', 'cli']), \
         patch('src.cli.run_cli') as mock_run_cli:
        main()
        mock_run_cli.assert_called_once()

def test_main_api_mode():
    with patch('sys.argv', ['cli.py', 'api']), \
         patch('builtins.print') as mock_print:
        main()
        mock_print.assert_called_with("API mode not implemented yet.")

def test_main_invalid_mode():
    with patch('sys.argv', ['cli.py', 'invalid']):
        with pytest.raises(SystemExit) as excinfo:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                main()
            assert excinfo.value.code == 2
            assert "cli.py: error: argument mode: invalid choice: 'invalid' (choose from 'cli', 'api')" in mock_stderr.getvalue()

def test_load_and_process_book(mock_book_data):
    mock_book_data.chunks = ['chunk1', 'chunk2']
    mock_book_data.embeddings = ['embedding1', 'embedding2']
    
    with patch('src.cli.load_and_preprocess_text') as mock_preprocess, \
         patch('src.cli.get_or_create_chunks_and_embeddings') as mock_get_embeddings:
        mock_preprocess.return_value = {'chunks': ['chunk1', 'chunk2']}
        mock_get_embeddings.return_value = mock_book_data
        
        result = load_and_process_book("Sample text content")
        
        assert isinstance(result, BookDataInterface)
        mock_preprocess.assert_called_once()

def test_answer_question(mock_book_data):
    with patch('src.cli.rag_query') as mock_rag_query:
        mock_rag_query.return_value = "Sample answer"
        
        result = answer_question("Sample question", mock_book_data)
        
        assert result == "Sample answer"
        mock_rag_query.assert_called_once_with("Sample question", mock_book_data)

@pytest.mark.parametrize("user_input", ["input1", "input2"])
def test_run_cli(user_input, mock_book_data):
    # Ensure mock_book_data has the necessary attributes
    mock_book_data.chunks = ['chunk1', 'chunk2']
    mock_book_data.embeddings = ['embedding1', 'embedding2']
    
    with patch('builtins.input', side_effect=[user_input]), \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.load_and_process_book') as mock_load_book, \
         patch('src.cli.rag_query') as mock_rag_query, \
         patch('builtins.print') as mock_print:
        mock_load_book.return_value = mock_book_data
        mock_rag_query.return_value = "Sample answer"
        
        run_cli()
        
        assert mock_load_book.call_count == 1
        assert mock_rag_query.call_count == 1
        assert mock_print.call_count >= 1

def test_run_cli_file_not_found():
    with patch('builtins.input', return_value='nonexistent_file.txt'), \
         patch('os.path.exists', return_value=False), \
         patch('builtins.print') as mock_print:
        
        run_cli()
        
        mock_print.assert_called_with("An error occurred: The file nonexistent_file.txt does not exist.")
