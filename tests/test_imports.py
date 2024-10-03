import pytest

def test_imports():
    import openai
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import tiktoken
    from dotenv import load_dotenv

    # If these imports succeed, the test passes

def test_config_import():
    from src.config import OPENAI_API_KEY, MAX_TOKENS, CHUNK_SIZE, OVERLAP

    assert OPENAI_API_KEY, "OPENAI_API_KEY should not be empty"
    assert isinstance(MAX_TOKENS, int), "MAX_TOKENS should be an integer"
    assert isinstance(CHUNK_SIZE, int), "CHUNK_SIZE should be an integer"
    assert isinstance(OVERLAP, int), "OVERLAP should be an integer"

def test_openai_api_key():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY should be set in the .env file"
