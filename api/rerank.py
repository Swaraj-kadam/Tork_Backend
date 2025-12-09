# api/rerank.py
import json
import re
from typing import List

from .utils.ollama_llm import ask_llama, DEFAULT_MODEL

def rerank_chunks_with_llama(
    query: str,
    chunks: List[str],
    top_k: int = 8,
    model: str = DEFAULT_MODEL,
) -> List[str]:
    """
    Use LLaMA (via Ollama) to rerank retrieved chunks by relevance to the query.
    Returns the top_k chunks in the new order.
    """

    if not chunks:
        return []

    # 1) Build numbered chunk list for the model
    numbered_chunks = []
    for idx, text in enumerate(chunks):
        # Truncate very long chunks to avoid context overflow
        short_text = text[:1500]
        numbered_chunks.append(f"[{idx}] {short_text}")
    chunks_block = "\n\n".join(numbered_chunks)

    # 2) Ask the model to select best chunk indices
    prompt = f"""
You are a search ranking assistant.

Your job is to choose which text chunks are MOST relevant to the user question.

QUESTION:
{query}

CHUNKS:
{chunks_block}

Return ONLY a JSON list of at most {top_k} integers (chunk indices) ordered from most to least relevant.
Example: [3, 0, 2]
Do not add any explanation.
"""

    raw_response = ask_llama(prompt, model=model).strip()

    # 3) Extract first JSON list from the response
    # Sometimes models wrap it in extra text; we pull out the [ ... ] part.
    match = re.search(r"\[(.*?)\]", raw_response, re.DOTALL)
    if not match:
        # Fallback: no parseable list — just return first top_k chunks as-is
        return chunks[:top_k]

    list_str = "[" + match.group(1) + "]"

    try:
        indices = json.loads(list_str)
        if not isinstance(indices, list):
            raise ValueError("parsed result is not a list")

        # Clean indices: must be ints within range
        valid_indices = [
            int(i) for i in indices
            if isinstance(i, int) and 0 <= int(i) < len(chunks)
        ]
        if not valid_indices:
            return chunks[:top_k]

        # Take top_k unique indices in order
        seen = set()
        ordered_indices = []
        for i in valid_indices:
            if i not in seen:
                ordered_indices.append(i)
                seen.add(i)
            if len(ordered_indices) >= top_k:
                break

        return [chunks[i] for i in ordered_indices]

    except Exception:
        # Parsing failed → graceful fallback
        return chunks[:top_k]
