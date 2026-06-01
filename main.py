"""
main.py
-------
The FastAPI server — the "orchestrator" that calls search → transcript → embed → rank
in the right order and exposes a clean HTTP API.

Endpoints:
  GET  /           → search UI (index.html)
  GET  /health     → health check (confirms server is running)
  POST /search     → main endpoint; accepts a JSON body, returns ranked videos
  GET  /search-get → convenience GET version (pass query as URL param, for browser testing)

How to run:
  uvicorn main:app --reload --host 127.0.0.1 --port 8000

Then open http://localhost:8000 in a browser for the search UI.
Health check: http://localhost:8000/health
API docs: http://localhost:8000/docs
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import traceback

from search import fetch_candidates
from transcript import process_all_videos
from embeddings import embed_text, embed_chunks
from ranker import rank

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube Semantic Search",
    description="Phase 1 MVP — semantic search over YouTube video transcripts",
    version="1.0.0",
)

# Allow the frontend (index.html opened as a local file) to call this API.
# In production you'd restrict origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_n: int = 5                    # how many videos to return (default 5, max sensible is 10)
    results_per_phrase: int = 4       # YouTube results fetched per expanded query phrase
    max_videos_to_probe: int = 8      # upper bound of videos to probe for transcripts
    min_transcript_chunks: int = 6    # stop probing once this many transcript chunks are collected


class ChunkResult(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str
    final_score: float
    best_chunk_score: float
    matched_snippet: str


class SearchResponse(BaseModel):
    query: str
    total_candidates: int
    total_chunks: int
    results: list[ChunkResult]
    warning: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    """Serve the search UI at the app root."""
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/health")
def health_check():
    """Quick check that the server is alive."""
    return {
        "status": "running",
        "message": "YouTube Semantic Search API - POST /search to query",
        "docs": "/docs",
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Main search endpoint.

    Pipeline:
      1. fetch_candidates  → YouTube Data API (search.py)
      2. process_all_videos → transcript download + chunking (transcript.py)
      3. embed_chunks      → sentence-transformer vectors (embeddings.py)
      4. embed_text        → embed the user query (embeddings.py)
      5. rank              → FAISS search + aggregation (ranker.py)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    print(f"\n{'='*60}")
    print(f"[main] New search: '{req.query}'")
    print(f"{'='*60}")

    try:
        # Step 1: Fetch candidate videos from YouTube
        candidates = fetch_candidates(
            req.query,
            results_per_phrase=req.results_per_phrase,
        )
        if not candidates:
            return SearchResponse(
                query=req.query,
                total_candidates=0,
                total_chunks=0,
                results=[],
            )

        # Step 2: Download transcripts and chunk them (metadata fallback if captions blocked)
        all_chunks, used_metadata = process_all_videos(
            candidates,
            min_chunks=max(1, req.min_transcript_chunks),
            max_videos_to_probe=max(1, req.max_videos_to_probe),
        )
        if not all_chunks:
            raise HTTPException(
                status_code=404,
                detail="No usable content found for any candidate video."
            )

        warning = None
        if used_metadata:
            warning = (
                "YouTube is rate-limiting caption downloads right now. "
                "Results are ranked from video titles and descriptions instead of full transcripts."
            )

        # Step 3: Embed all chunks
        chunk_vectors, all_chunks = embed_chunks(all_chunks)

        # Step 4: Embed the user query
        query_vector = embed_text(req.query)

        # Step 5: Rank and return top_n videos
        ranked = rank(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            chunks=all_chunks,
            top_n=req.top_n,
        )

        return SearchResponse(
            query=req.query,
            total_candidates=len(candidates),
            total_chunks=len(all_chunks),
            results=[ChunkResult(**r) for r in ranked],
            warning=warning,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[main] ERROR:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-get")
def search_get(query: str, top_n: int = 5):
    """
    GET version for easy browser/curl testing.
    Usage: http://localhost:8000/search-get?query=sentiment+analysis+flask
    """
    return search(SearchRequest(query=query, top_n=top_n))
