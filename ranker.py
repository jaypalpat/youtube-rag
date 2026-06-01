"""
ranker.py
---------
Two responsibilities:
  1. FAISS index — store all chunk vectors and find the top-K most similar
                   to the query vector in milliseconds.
  2. Score aggregation — each video has many chunks. We combine per-chunk
                         scores into one per-video score before returning results.

What is FAISS?
  FAISS (Facebook AI Similarity Search) is a library that can search through
  millions of vectors extremely fast. For our MVP with ~300 chunks it's instant,
  but it would still handle 1 million chunks comfortably on a laptop.

How similarity works here:
  Because we L2-normalised our embeddings in embeddings.py,
  cosine similarity = dot product. FAISS's IndexFlatIP (inner product) therefore
  gives us cosine similarity scores in [-1, 1].
  Score of 1.0 means identical. Score of 0.0 means unrelated. Score < 0 is rare.

Score aggregation strategy:
  A video like "Complete NLP Course" might have 40 chunks. One chunk perfectly
  matches "deploy sentiment model to Flask". We don't want to rank this video
  below a 2-chunk video that had a mediocre match in both chunks.

  We use: final_score = 0.7 × best_chunk_score + 0.3 × avg_top3_score
  This rewards videos that have at least one excellent match (the 0.7 part)
  while slightly preferring videos that are consistently relevant (0.3 part).
  You can tune these weights later.
"""

import numpy as np
import faiss
from collections import defaultdict


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat inner-product index from the chunk vectors.

    IndexFlatIP is the simplest FAISS index — it does exact brute-force search.
    For <10,000 vectors this is fine and gives perfect results.
    (For 1M+ vectors you'd switch to IndexIVFFlat or HNSW for speed.)

    vectors: shape (num_chunks, embedding_dim), float32
    Returns a FAISS index ready to be queried.
    """
    dim = vectors.shape[1]   # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"[ranker] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def search_index(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the top-K most similar chunks.

    query_vector: 1-D array of shape (384,)
    k           : how many top chunks to retrieve before aggregating

    Returns:
        scores  : 1-D array of similarity scores for top-K chunks
        indices : 1-D array of chunk indices into the original chunks list
    """
    # FAISS expects shape (1, dim) for a single query
    q = query_vector.reshape(1, -1).astype("float32")
    scores, indices = index.search(q, k)
    return scores[0], indices[0]


def aggregate_scores(
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Convert per-chunk scores into per-video scores and return the top_n videos.

    For each video we collect all the chunk scores from our top-K results,
    then combine them: 70% best chunk + 30% average of best 3.

    Returns a list of result dicts, sorted by final_score descending.
    Each result dict:
    {
        "video_id"        : "abc123",
        "title"           : "Build NLP Apps with Flask",
        "url"             : "https://www.youtube.com/watch?v=abc123",
        "channel"         : "Code With Tim",
        "final_score"     : 0.82,
        "best_chunk_score": 0.91,
        "matched_snippet" : "In this section we deploy our sentiment model...",
    }
    """
    # Group scores by video_id
    video_chunk_scores: dict[str, list] = defaultdict(list)
    video_meta: dict[str, dict] = {}

    for score, idx in zip(top_scores, top_indices):
        if idx == -1:   # FAISS returns -1 when fewer than k results exist
            continue
        chunk = chunks[idx]
        vid = chunk["video_id"]
        video_chunk_scores[vid].append((float(score), chunk["text"]))
        if vid not in video_meta:
            video_meta[vid] = {
                "video_id": vid,
                "title"   : chunk["title"],
                "url"     : chunk["url"],
                "channel" : chunk["channel"],
            }

    results = []
    for vid, score_text_pairs in video_chunk_scores.items():
        # Sort chunks for this video by score, highest first
        score_text_pairs.sort(key=lambda x: x[0], reverse=True)
        chunk_scores = [s for s, _ in score_text_pairs]

        best_score = chunk_scores[0]
        top3_avg   = float(np.mean(chunk_scores[:3]))
        final      = 0.7 * best_score + 0.3 * top3_avg
        snippet    = score_text_pairs[0][1][:300]   # first 300 chars of best chunk

        results.append({
            **video_meta[vid],
            "final_score"     : round(final, 4),
            "best_chunk_score": round(best_score, 4),
            "matched_snippet" : snippet,
        })

    # Sort by final score, return top_n
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]


def rank(
    query_vector: np.ndarray,
    chunk_vectors: np.ndarray,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Full ranking pipeline: build index → search → aggregate → return top_n.

    This is the function main.py calls.

    query_vector  : embedded user query, shape (384,)
    chunk_vectors : all embedded transcript chunks, shape (num_chunks, 384)
    chunks        : list of chunk dicts (parallel to chunk_vectors)
    top_n         : how many final video results to return
    """
    index = build_index(chunk_vectors)
    top_scores, top_indices = search_index(index, query_vector, k=min(50, len(chunks)))
    results = aggregate_scores(top_scores, top_indices, chunks, top_n=top_n)
    print(f"[ranker] Returning top {len(results)} results.\n")
    return results