# src/file_processor.py
import os
from PyPDF2 import PdfReader
from docx import Document
from odf import text
from odf.opendocument import load
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    def process_file(self, file_path: str) -> str:
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        processors = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.odt': self._process_odt,
            '.txt': self._process_txt
        }

        processor = processors.get(extension)
        if processor:
            return processor(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _process_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            return '\n'.join(page.extract_text() for page in reader.pages)

    def _process_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

    def _process_odt(self, file_path: str) -> str:
        textdoc = load(file_path)
        allparas = textdoc.getElementsByType(text.P)
        return '\n'.join(str(p) for p in allparas)

    def _process_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()