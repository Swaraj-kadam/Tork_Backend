# api/embedding_cache.py
import json, os
from hashlib import sha256

CACHE_FILE = "embedding_cache.json"

# Load cache from disk
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        EMB_CACHE = json.load(f)
else:
    EMB_CACHE = {}

def get_cached_embedding(text: str):
    key = sha256(text.encode()).hexdigest()
    return EMB_CACHE.get(key)

def save_embedding(text: str, embedding):
    key = sha256(text.encode()).hexdigest()
    EMB_CACHE[key] = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    with open(CACHE_FILE, "w") as f:
        json.dump(EMB_CACHE, f)
