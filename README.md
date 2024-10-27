# RAG Book Question-Answering System

A Retrieval-Augmented Generation (RAG) system for intelligent book analysis and question answering, built with Flask and OpenAI. The system processes documents, creates embeddings, and uses RAG to generate accurate answers to questions about the uploaded texts.

## Features

- **Document Processing**: Support for multiple formats (PDF, DOCX, TXT, ODT)
- **Real-time Progress**: WebSocket-based progress updates during processing
- **Advanced Text Analysis**: 
  - Named entity recognition
  - Date extraction
  - Key phrase identification
- **Hybrid Search**: 
  - Semantic search using embeddings
  - Lexical search with BM25
  - Query expansion with synonyms
- **Caching System**: Efficient storage for embeddings and responses
- **Web Interface**: Clean UI with Markdown support for responses
- **CLI Interface**: Command-line interface for direct interaction

## Technical Stack

### Backend
- Flask + Flask-SocketIO for web server
- OpenAI API for embeddings and text generation
- Pinecone for vector storage
- NLTK & spaCy for text processing

### Frontend
- jQuery for AJAX requests
- Marked.js for Markdown rendering
- Font Awesome for icons
- WebSocket for real-time updates

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   FLASK_SECRET_KEY=your_flask_secret_key
   ADMIN_PASSWORD=your_admin_password
   TESTER_PASSWORD=your_tester_password
   ```
5. Run the application:
   ```bash
   flask run
   ```
6. Access the web interface at `http://localhost:5000`

## Usage

- Upload documents through the web interface to start processing.
- Use the CLI for direct interaction and testing of the system's capabilities.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
