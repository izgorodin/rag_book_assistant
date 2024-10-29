# src/file_processor.py
import os
from typing import Optional
import pypdf
from docx import Document
from odf import text, teletype
from odf.opendocument import load
from src.utils.logger import get_main_logger, get_rag_logger

# Initialize loggers for main application and RAG processing
logger = get_main_logger()
rag_logger = get_rag_logger()

class FileProcessor:
    def process_file(self, file_path: str) -> Optional[str]:
        """Process different file types and return text content."""
        # Extract the file extension from the file path
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Mapping of file extensions to their respective processing methods
        processors = {
            '.txt': self._process_txt,
            '.pdf': self._process_pdf,
            '.doc': self._process_doc,
            '.docx': self._process_docx,
            '.odt': self._process_odt
        }

        # Get the appropriate processor based on the file extension
        processor = processors.get(ext)
        if not processor:
            # Log and raise an error if the file type is unsupported
            error_msg = f"Unsupported file type: {ext}"
            logger.error(error_msg)
            rag_logger.error(f"\nFile Processing Error:\n{error_msg}\n{'-'*50}")
            raise ValueError(error_msg)

        try:
            # Log the start of file processing
            logger.info(f"Processing {ext} file: {file_path}")
            content = processor(file_path)  # Call the appropriate processing method
            logger.info(f"Successfully processed file, content length: {len(content)}")
            rag_logger.info(
                f"\nFile Processing:\n"
                f"File: {file_path}\n"
                f"Type: {ext}\n"
                f"Content length: {len(content)} chars\n"
                f"{'-'*50}"
            )
            return content  # Return the processed content
        except Exception as e:
            # Log and raise an error if processing fails
            error_msg = f"Error processing {ext} file: {str(e)}"
            logger.error(error_msg)
            rag_logger.error(f"\nFile Processing Error:\n{error_msg}\n{'-'*50}")
            raise

    def _process_txt(self, file_path: str) -> str:
        """Process a .txt file and return its content as a string."""
        encodings = ['utf-8', 'latin-1', 'cp1251', 'ascii']  # List of encodings to try
        for encoding in encodings:
            try:
                # Attempt to open and read the file with the specified encoding
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()  # Return the file content if successful
            except UnicodeDecodeError:
                continue  # Try the next encoding if a decode error occurs
        # Raise an error if none of the encodings work
        raise UnicodeDecodeError(f"Could not decode file with any of: {encodings}")

    def _process_pdf(self, file_path: str) -> str:
        """Process a .pdf file and return its text content."""
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)  # Read the PDF file
            # Extract text from each page and join them into a single string
            return ' '.join(page.extract_text() for page in reader.pages)

    def _process_docx(self, file_path: str) -> str:
        """Process a .docx file and return its text content."""
        doc = Document(file_path)  # Load the .docx file
        # Join the text from all paragraphs into a single string
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

    def _process_odt(self, file_path: str) -> str:
        """Process a .odt file and return its text content."""
        textdoc = load(file_path)  # Load the .odt file
        allparas = textdoc.getElementsByType(text.P)  # Get all paragraphs
        # Extract and join text from all paragraphs
        return '\n'.join(teletype.extractText(para) for para in allparas)

    def _process_doc(self, file_path: str) -> str:
        """Process a .doc file (currently not supported)."""
        # Raise an error indicating that .doc format is not supported
        raise NotImplementedError("DOC format not supported yet")
