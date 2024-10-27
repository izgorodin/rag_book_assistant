import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"  # Ensure this matches the desired model
MAX_TOKENS = 15000
CHUNK_SIZE = 300
OVERLAP = 150
TOP_K_CHUNKS = 15  # Added for clarity


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east1-gcp"
EMBEDDING_DIMENSION = 1536  # This should match your embedding model's output dimension
PINECONE_INDEX_NAME = "book-embeddings"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"


CACHE_DIR = 'data/cache'
EMBEDDINGS_DIR = 'data/embeddings'
BATCH_SIZE = 100