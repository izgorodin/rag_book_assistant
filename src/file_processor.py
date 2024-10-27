# src/file_processor.py
import os
from pypdf import PdfReader
from docx import Document
from odf import text
from odf.opendocument import load
from src.logger import setup_logger

logger = setup_logger()

class FileProcessor:
    def process_file(self, file_path: str) -> str:
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        logger.info(f"Processing file: {file_path} with extension: {extension}")

        try:
            processors = {
                '.pdf': self._process_pdf,
                '.docx': self._process_docx,
                '.odt': self._process_odt,
                '.txt': self._process_txt
            }

            processor = processors.get(extension)
            if processor:
                logger.info(f"Using processor for extension: {extension}")
                return processor(file_path)
            else:
                logger.error(f"Unsupported file format: {extension}")
                raise ValueError(f"Unsupported file format: {extension}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def _process_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = '\n'.join(page.extract_text() for page in reader.pages)
                return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            raise

    def _process_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX file: {str(e)}")
            raise

    def _process_odt(self, file_path: str) -> str:
        try:
            textdoc = load(file_path)
            allparas = textdoc.getElementsByType(text.P)
            text = '\n'.join(str(p) for p in allparas)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing ODT file: {str(e)}")
            raise

    def _process_txt(self, file_path: str) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1251', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    return text.strip()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error processing TXT file: {str(e)}")
                raise
                
        raise UnicodeDecodeError(f"Unable to decode file with any of the encodings: {encodings}")
