"""
embeddings.py
-------------
Converts text into numerical vectors (embeddings) using a local sentence-transformer model.

What is an embedding?
  Text → a list of ~384 numbers (floats).
  Similar sentences end up with similar number lists.
  "How to build a flask API" and "deploying Python web apps" will have vectors
  that are close together in space, even though they share no exact words.
  This is the core of semantic search.

Why sentence-transformers?
  - Runs locally on your machine, no API key needed.
  - "all-MiniLM-L6-v2" is fast, small (80MB), and surprisingly good.
  - First run downloads the model (~80MB). Subsequent runs use the cached copy.

Model choice:
  - "all-MiniLM-L6-v2"  : fast, small, great for prototyping  ← we use this
  - "all-mpnet-base-v2" : slower but more accurate, drop-in replacement
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once at module import time.
# This takes ~2-5 seconds the first time (downloads + loads weights).
# After that it's instant because the model is cached on disk.
print("[embeddings] Loading sentence-transformer model (first run downloads ~80MB)...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[embeddings] Model loaded.\n")


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single piece of text. Returns a 1-D numpy array of shape (384,).

    Example:
        vec = embed_text("How do I train a neural network?")
        # vec is [0.023, -0.114, 0.881, ...] — 384 numbers
    """
    return MODEL.encode(text, normalize_embeddings=True)


def embed_batch(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts efficiently in batches.
    Returns a 2-D numpy array of shape (len(texts), 384).

    Processing in batches is much faster than one-at-a-time because the GPU/CPU
    can work on many texts in parallel.

    batch_size     : how many texts to process at once. 64 is a safe default.
    show_progress  : prints a progress bar if True (useful for 100+ chunks).
    """
    embeddings = MODEL.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # L2-normalise so cosine similarity = dot product
        show_progress_bar=show_progress,
    )
    return np.array(embeddings, dtype="float32")


def embed_chunks(chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Take a list of chunk dicts (from transcript.py) and embed the "text" field
    of each chunk.

    Returns:
        vectors : np.ndarray of shape (num_chunks, 384)  — the embeddings
        chunks  : the same list of chunk dicts (unchanged), so you can zip them
                  with the vectors to know which vector belongs to which chunk.

    Example:
        vectors, chunks = embed_chunks(all_chunks)
        # vectors[42] is the embedding for chunks[42]["text"]
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"[embeddings] Embedding {len(texts)} chunks...")
    vectors = embed_batch(texts)
    print(f"[embeddings] Done. Vector matrix shape: {vectors.shape}\n")
    return vectors, chunks