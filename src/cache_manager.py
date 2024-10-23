import os
import shutil
import hashlib
import pickle

CACHE_DIR = 'data/cache'
EMBEDDINGS_DIR = 'data/embeddings'
WATCHED_FILES = ['src/cli.py', 'src/rag.py', 'src/embedding.py']

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def save_file_hashes():
    hashes = {}
    for file in WATCHED_FILES:
        if os.path.exists(file):
            hashes[file] = get_file_hash(file)
    
    with open('file_hashes.txt', 'w') as f:
        for file, hash_value in hashes.items():
            f.write(f"{file}:{hash_value}\n")

def check_for_changes():
    if not os.path.exists('file_hashes.txt'):
        return True
    
    with open('file_hashes.txt', 'r') as f:
        stored_hashes = dict(line.strip().split(':') for line in f)
    
    for file in WATCHED_FILES:
        if os.path.exists(file):
            current_hash = get_file_hash(file)
            if file not in stored_hashes or stored_hashes[file] != current_hash:
                return True
    
    return False

def clear_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
    if os.path.exists(EMBEDDINGS_DIR):
        shutil.rmtree(EMBEDDINGS_DIR)
        os.makedirs(EMBEDDINGS_DIR)
    print("Cache cleared due to changes in key files.")

def manage_cache():
    if check_for_changes():
        clear_cache()
        save_file_hashes()

# New functions

def get_cache_key(text: str) -> str:
    """
    Generate a unique cache key for the given text.
    
    Args:
        text (str): The input text to generate a key for.
    
    Returns:
        str: A unique cache key.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_to_cache(key: str, data: any):
    """
    Save data to the cache using a hashed key.

    Args:
        key (str): The cache key.
        data (any): The data to be cached.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    hashed_key = hashlib.md5(key.encode()).hexdigest()
    file_path = os.path.join(CACHE_DIR, f"{hashed_key}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(key: str) -> any:
    """
    Load data from the cache using the given key.
    
    Args:
        key (str): The cache key.
    
    Returns:
        any: The cached data if found, None otherwise.
    """
    file_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None
