import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"  # Ensure this matches the desired model
MAX_TOKENS = 15000


# Chunking Configuration
CHUNK_SIZE = 1000
OVERLAP = 150
TOP_K_CHUNKS = 10  # Added for clarity

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east1-gcp"
EMBEDDING_DIMENSION = 1536  # This should match your embedding model's output dimension
PINECONE_INDEX_NAME = "book-embeddings"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"


# Cache and Embeddings Directory Configuration
CACHE_DIR = 'data/cache'
EMBEDDINGS_DIR = 'data/embeddings'

# Batch Size Configuration
PINECONE_BATCH_SIZE = 100

# Flask Secret Key Configuration
FLASK_SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') 
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
TESTER_PASSWORD = os.environ.get('TESTER_PASSWORD')

# OpenAI HTTP Client Configuration
OPENAI_HTTP_CONFIG = {
    'timeout': 30.0,  # Request timeout in seconds
    'max_retries': 3,  # Maximum number of retry attempts
    'max_keepalive_connections': 5,  # Maximum number of keepalive connections
    'max_connections': 10,  # Maximum number of connections
    'backoff_factor': 0.5,  # Exponential backoff factor for retries
    'retry_statuses': [408, 429, 500, 502, 503, 504]  # HTTP status codes to retry on
}

# Configuration for file uploads
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')

# Validate Firebase configuration
if not FIREBASE_CREDENTIALS_PATH:
    raise ValueError("FIREBASE_CREDENTIALS_PATH environment variable is not set")
if not FIREBASE_STORAGE_BUCKET:
    raise ValueError("FIREBASE_STORAGE_BUCKET environment variable is not set")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)