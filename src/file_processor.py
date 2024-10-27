# src/file_processor.py
import os
from typing import Optional
import PyPDF2
from docx import Document
from odf import text, teletype
from odf.opendocument import load
from logger import setup_logger

logger = setup_logger()

class FileProcessor:
    def process_file(self, file_path: str) -> Optional[str]:
        """Process different file types and return text content."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        processors = {
            '.txt': self._process_txt,
            '.pdf': self._process_pdf,
            '.doc': self._process_doc,
            '.docx': self._process_docx,
            '.odt': self._process_odt
        }

        processor = processors.get(ext)
        if not processor:
            raise ValueError(f"Unsupported file type: {ext}")

        try:
            return processor(file_path)
        except Exception as e:
            logger.error(f"Error processing {ext} file: {str(e)}")
            raise

    def _process_txt(self, file_path: str) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1251', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Could not decode file with any of: {encodings}")

    def _process_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)

    def _process_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

    def _process_odt(self, file_path: str) -> str:
        textdoc = load(file_path)
        allparas = textdoc.getElementsByType(text.P)
        return '\n'.join(teletype.extractText(para) for para in allparas)

    def _process_doc(self, file_path: str) -> str:
        # Для .doc файлов нужно использовать дополнительные библиотеки
        # или конвертировать в .docx
        raise NotImplementedError("DOC format not supported yet")
