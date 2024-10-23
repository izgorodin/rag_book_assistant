# RAG System Architecture

## Overview

The RAG (Retrieval-Augmented Generation) system is designed to process large text documents, such as books, and answer user queries based on the content. It combines advanced natural language processing techniques with OpenAI's GPT model to provide accurate and contextually relevant answers.

## System Components

### 1. Entry Points

#### 1.1 Command-Line Interface (CLI)
- File: `src/cli.py`
- Main function: `run_cli()`
- Handles user interactions for loading books and processing queries via command line.

#### 1.2 Web Application
- File: `app.py`
- Main class: Flask application
- Provides a web interface for uploading books and asking questions.

### 2. Core Components

#### 2.1 File Processing
- File: `src/file_processor.py`
- Main class: `FileProcessor`
- Responsible for reading and extracting text from various file formats (PDF, DOCX, ODT, TXT).

#### 2.2 Text Processing
- File: `src/text_processing.py`
- Key functions: `load_and_preprocess_text()`, `split_into_chunks()`
- Handles preprocessing of text, including splitting into manageable chunks and extracting metadata.

#### 2.3 Embedding Generation
- File: `src/embedding.py`
- Key functions: `create_embeddings()`, `get_or_create_chunks_and_embeddings()`
- Manages the creation and caching of text embeddings using OpenAI's API.

#### 2.4 RAG Query Processing
- File: `src/rag.py`
- Main function: `rag_query()`
- Implements the core RAG functionality, including finding relevant chunks and generating answers.

#### 2.5 Book Data Interface
- File: `src/book_data_interface.py`
- Main class: `BookDataInterface`
- Provides a unified interface for storing and accessing book data.

### 3. Utility Components

#### 3.1 Logging
- File: `src/logger_config.py`
- Main function: `setup_logger()`
- Configures logging for the entire application.

## Data Flow

1. User inputs a book file (via CLI or web interface).
2. `FileProcessor` extracts text from the file.
3. Text is preprocessed and split into chunks.
4. Embeddings are generated for each chunk and metadata is extracted.
5. Processed data is stored in a `BookDataInterface` object.
6. User submits a query.
7. RAG system finds relevant chunks using hybrid search.
8. Context is constructed from relevant chunks and metadata.
9. An answer is generated using OpenAI's GPT model.
10. Answer is returned to the user.

## Future Improvements

1. Implement more advanced caching strategies for embeddings and processed data.
2. Enhance the web interface with real-time processing updates.
3. Add support for multiple books and context switching.
4. Implement user authentication and session management for the web application.
5. Optimize performance for very large documents.

## Project Structure

rag_book_assistant/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── cli.py
│   ├── api.py
│   ├── config.py
│   ├── logger.py
│   ├── exceptions.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── book_data.py
│   │   └── file_processor.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── text_processor.py
│   │   └── embedding_creator.py
│   ├── search/
│   │   ├── __init__.py
│   │   └── hybrid_search.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── query_processor.py
│   └── utils/
│       ├── __init__.py
│       └── cache_manager.py
│
├── app.py
│
├── tests/
│   ├── __init__.py
│   ├── test_file_processor.py
│   ├── test_text_processor.py
│   ├── test_embedding_creator.py
│   ├── test_hybrid_search.py
│   └── test_query_processor.py
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   └── api_documentation.md
│
├── data/
│   ├── cache/
│   └── embeddings/
│
├── logs/
│
├── web/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md