import requests
import numpy as np
import logging
import json 
import  time

from api.embedding_cache import get_cached_embedding, save_embedding

logger = logging.getLogger(__name__)

# Base URLs for Ollama
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Default model names
DEFAULT_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

CURRENT_MODEL = DEFAULT_MODEL

def set_model(model_name):
    global CURRENT_MODEL
    CURRENT_MODEL = model_name

# -------------------------------------------------
# 🧠 1. Generate embeddings for text chunks
# -------------------------------------------------
def generate_embeddings(chunks, model: str = EMBED_MODEL):
    """
    Generate embeddings for a list of text chunks using Ollama's embedding model.
    Returns a list of NumPy arrays.
    """
    embeddings = []
    for chunk in chunks:
        payload = {"model": model, "prompt": chunk}
        try:
            res = requests.post(OLLAMA_EMBED_URL, json=payload)
            data = res.json()
            emb = np.array(data.get("embedding", []))
            embeddings.append(emb)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            embeddings.append(np.zeros(1536))  # fallback vector
    return embeddings




# -------------------------------------------------
# 🧮 2. Generate query embedding
# -------------------------------------------------
# def generate_query_embedding(query: str, model: str = EMBED_MODEL):
#     try:
#         payload = {"model": model, "prompt": query}
#         res = requests.post(OLLAMA_EMBED_URL, json=payload)
#         data = res.json()
#         emb = np.array(data.get("embedding", []))
#         return emb
#     except Exception as e:
#         logger.error(f"Query embedding failed: {e}")
#         return np.zeros(1536)

def generate_query_embedding(query: str, model: str = EMBED_MODEL):
    cached = get_cached_embedding(query)
    if cached:
        return np.array(cached)

    payload = {"model": model, "prompt": query}
    try:
        res = requests.post(OLLAMA_EMBED_URL, json=payload)
        data = res.json()

        emb = np.array(data.get("embedding", []))

        # ❗ Stop empty embeddings
        if emb.size == 0:
            logger.error("Embedding model returned EMPTY embedding!")
            return None

        save_embedding(query, emb)
        return emb

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None

    

# -------------------------------------------------
# 💬 3. Ask LLaMA for answer (non-streaming)
# -------------------------------------------------
def ask_llama(prompt: str, model: str = DEFAULT_MODEL):
    """
    Sends prompt to LLaMA model via Ollama and returns complete response.
    """

    if model is None:
        model = CURRENT_MODEL
        
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        start = time.time()
        res = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
        data = res.json()
        print(f"✅ LLaMA response in {time.time()-start:.2f}s")
        return data.get("response", "Error: no response from model.")
    except Exception as e:
        logger.error(f"LLaMA request failed: {e}")
        return "Error: model not available."


# -------------------------------------------------
# 📦 4. List all available models
# -------------------------------------------------
def list_models():
    """
    Returns list of all downloaded Ollama models.
    """
    try:
        res = requests.get(OLLAMA_TAGS_URL)
        data = res.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return [DEFAULT_MODEL]
    
# ✅ Streaming version
def stream_llama(prompt: str, model: str = DEFAULT_MODEL):
    """
    Streams LLaMA model output token by token from Ollama.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}

    try:
        with requests.post(OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=600) as res:
            for line in res.iter_lines():
                if line:
                    try:
                        # Decode JSON safely
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if token:
                            yield token
                    except Exception:
                        continue
    except Exception as e:
        logger.error(f"LLaMA streaming failed: {e}")
        yield "Error: model not available."
