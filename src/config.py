import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"
MAX_TOKENS = 150
CHUNK_SIZE = 1000
OVERLAP = 200