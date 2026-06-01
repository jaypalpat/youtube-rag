# YouTube Semantic Search (RAG Pipeline)

Semantic search over YouTube video transcripts: fetch candidates, download transcripts, embed with sentence-transformers, and rank with FAISS.

**No YouTube API key required** — video search uses [yt-dlp](https://github.com/yt-dlp/yt-dlp); transcripts use [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api).

## Prerequisites

- Python 3.12+

## Setup

```bash
# From the repository root (this folder)

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | Search UI |
| http://localhost:8000/health | API health check (JSON) |
| http://localhost:8000/docs | OpenAPI / Swagger UI |

## Test the API

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sentiment analysis flask deployment",
    "top_n": 5,
    "results_per_phrase": 4,
    "max_videos_to_probe": 8,
    "min_transcript_chunks": 6
  }'
```

## Rate-limit tuning knobs

`transcript.py` now includes a local transcript cache and adaptive retry pacing.

- `YT_REQUEST_DELAY_SEC` (default: `1.0`) - base delay between requests
- `YT_MAX_REQUEST_DELAY_SEC` (default: `16.0`) - max backoff delay
- `YT_RATE_LIMIT_COOLDOWN_SEC` (default: `30.0`) - cooldown applied after rate-limit errors
- `YT_TRANSCRIPT_ATTEMPTS` (default: `3`) - retry attempts per video
- `YT_TRANSCRIPT_CACHE_DIR` (default: `./.cache/transcripts`) - transcript cache location

Example:

```bash
export YT_REQUEST_DELAY_SEC=1.5
export YT_MAX_REQUEST_DELAY_SEC=20
export YT_RATE_LIMIT_COOLDOWN_SEC=45
export YT_TRANSCRIPT_ATTEMPTS=3
```

## Known issue (rate limiting)

These changes were introduced after repeated `youtube-transcript-api` rate-limit failures during multiple runs. In other words, transcripts could be obtained on the first one or two runs. However, on subsequent runs, youtube started blocking requests. As a result, data collection fell back to metadata.

***Current status:*** even after adding adaptive backoff, cooldown, cache, and reduced probe fanout, we did **not** observe meaningful improvement in the rate-limit behavior yet. This area needs more work.

## Project layout

- `main.py` — FastAPI app and routes
- `search.py` — YouTube search (yt-dlp) and query expansion
- `transcript.py` — Transcript download and chunking
- `embeddings.py` — Sentence-transformer embeddings
- `ranker.py` — FAISS ranking
- `index.html` — Frontend (also served at `/`)
