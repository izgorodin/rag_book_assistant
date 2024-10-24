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
    with patch('sys.argv', ['cli.py', 'invalid']), \
         patch('builtins.print') as mock_print:
        main()
        mock_print.assert_called_with("Unknown mode: invalid")

def test_load_and_process_book():
    with patch('src.cli.load_and_preprocess_text') as mock_preprocess, \
         patch('src.cli.get_or_create_chunks_and_embeddings') as mock_get_embeddings:
        mock_preprocess.return_value = {'chunks': ['chunk1', 'chunk2']}
        mock_get_embeddings.return_value = mock_book_data()
        
        result = load_and_process_book("Sample text content")
        
        assert isinstance(result, BookDataInterface)
        mock_preprocess.assert_called_once()
        mock_get_embeddings.assert_called_once()

def test_answer_question(mock_book_data):
    with patch('src.cli.rag_query') as mock_rag_query:
        mock_rag_query.return_value = "Sample answer"
        
        result = answer_question("Sample question", mock_book_data)
        
        assert result == "Sample answer"
        mock_rag_query.assert_called_once_with("Sample question", mock_book_data)

@pytest.mark.parametrize("user_input,expected_calls", [
    (['path/to/book.txt', 'question1', 'exit'], 1),
    (['path/to/book.txt', 'question1', 'question2', 'exit'], 2),
])
def test_run_cli(user_input, expected_calls):
    with patch('builtins.input', side_effect=user_input), \
         patch('src.cli.load_and_process_book') as mock_load_book, \
         patch('src.cli.rag_query') as mock_rag_query, \
         patch('builtins.print') as mock_print:
        
        mock_load_book.return_value = mock_book_data()
        mock_rag_query.return_value = "Sample answer"
        
        run_cli()
        
        assert mock_load_book.call_count == 1
        assert mock_rag_query.call_count == expected_calls
        assert mock_print.call_count >= expected_calls  # At least one print per question

def test_run_cli_file_not_found():
    with patch('builtins.input', return_value='nonexistent_file.txt'), \
         patch('os.path.exists', return_value=False), \
         patch('builtins.print') as mock_print:
        
        run_cli()
        
        mock_print.assert_called_with("An error occurred: The file nonexistent_file.txt does not exist.")
