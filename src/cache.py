import os
import json
import hashlib
from typing import List, Dict, Any

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')

def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def save_to_cache(key: str, data: List[List[float]]) -> None:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def load_from_cache(key: str) -> List[List[float]]:
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

