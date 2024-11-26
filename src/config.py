import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"  # Ensure this matches the desired model
MAX_TOKENS = 15000

# Text Processing
CHUNK_SIZE = 1000
OVERLAP = 150
TOP_K_CHUNKS = 10

# Vector Store Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east1-gcp"
EMBEDDING_DIMENSION = 1536  # This should match your embedding model's output dimension
PINECONE_INDEX_NAME = "book-embeddings"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"
PINECONE_BATCH_SIZE = 200

# Storage Configuration
CACHE_DIR = 'data/cache'
EMBEDDINGS_DIR = 'data/embeddings'
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

# Authentication
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
TESTER_PASSWORD = os.environ.get('TESTER_PASSWORD')

# HTTP Client Configuration
OPENAI_HTTP_CONFIG = {
    'timeout': 30.0,
    'max_retries': 3,
    'max_keepalive_connections': 5,
    'max_connections': 10,
    'backoff_factor': 0.5,
    'retry_statuses': [408, 429, 500, 502, 503, 504]
}

# Batch Processing Configuration
BATCH_SIZES = {
    'embeddings': 200,      # OpenAI embeddings batch size
    'pinecone': 200,       # Pinecone upsert batch size
    'text_chunks': 1000,   # Text chunking size
    'analysis': 500        # Text analysis batch size
}

BATCH_SETTINGS = {
    'max_workers': 8,
    'chunk_overlap': 150,
    'retry_attempts': 3,
    'timeout': 30
}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)