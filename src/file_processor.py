# src/file_processor.py
import os
from typing import Dict, Callable
from pypdf import PdfReader
from docx import Document
from odf import text
from odf.opendocument import load
from src.logger import setup_logger
from src.error_handler import handle_rag_error, RAGError, DataSourceError
from src.types import Chunk

logger = setup_logger()

class FileProcessor:
    """
    A class for processing various file types and extracting their text content.
    Supports PDF, DOCX, ODT, and TXT file formats.
    """

    @handle_rag_error
    def process_file(self, file_path: str) -> Chunk:
        """
        Process a file and extract its text content based on the file extension.

        Args:
            file_path (str): The path to the file to be processed.

        Returns:
            Chunk: The extracted text content from the file.

        Raises:
            DataSourceError: If the file format is unsupported or there's an issue processing the file.
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        logger.info(f"Processing file: {file_path} with extension: {extension}")

        processors: Dict[str, Callable[[str], str]] = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.odt': self._process_odt,
            '.txt': self._process_txt
        }

        processor = processors.get(extension)
        if processor:
            logger.info(f"Using processor for extension: {extension}")
            return Chunk(processor(file_path))
        else:
            logger.error(f"Unsupported file format: {extension}")
            raise DataSourceError(f"Unsupported file format: {extension}")

    def _process_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content.

        Raises:
            DataSourceError: If there's an issue reading or processing the PDF file.
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                return '\n'.join(page.extract_text() for page in reader.pages)
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            raise DataSourceError(f"Error processing PDF file: {str(e)}")

    def _process_docx(self, file_path: str) -> str:
        """
        Extract text content from a DOCX file.

        Args:
            file_path (str): The path to the DOCX file.

        Returns:
            str: The extracted text content.

        Raises:
            DataSourceError: If there's an issue reading or processing the DOCX file.
        """
        try:
            doc = Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
            raise DataSourceError(f"Error processing DOCX file: {str(e)}")

    def _process_odt(self, file_path: str) -> str:
        """
        Extract text content from an ODT file.

        Args:
            file_path (str): The path to the ODT file.

        Returns:
            str: The extracted text content.

        Raises:
            DataSourceError: If there's an issue reading or processing the ODT file.
        """
        try:
            textdoc = load(file_path)
            allparas = textdoc.getElementsByType(text.P)
            return '\n'.join(str(p) for p in allparas)
        except Exception as e:
            logger.error(f"Error processing ODT file {file_path}: {str(e)}")
            raise DataSourceError(f"Error processing ODT file: {str(e)}")

    def _process_txt(self, file_path: str) -> str:
        """
        Extract text content from a TXT file.

        Args:
            file_path (str): The path to the TXT file.

        Returns:
            str: The extracted text content.

        Raises:
            DataSourceError: If there's an issue reading or processing the TXT file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error processing TXT file {file_path}: {str(e)}")
            raise DataSourceError(f"Error processing TXT file: {str(e)}")
