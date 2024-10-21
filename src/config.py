import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"  # Ensure this matches the desired model
MAX_TOKENS = 10000
CHUNK_SIZE = 500
OVERLAP = 200
TOP_K_CHUNKS = 5  # Added for clarity