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
TOP_K_CHUNKS = 15
USE_CACHING = os.getenv("USE_CACHING", "False").lower() == "false"  # Добавляем опцию кэширования
UPDATE_EMBEDDINGS = True
