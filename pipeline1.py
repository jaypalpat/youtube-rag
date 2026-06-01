"""
pipeline1.py
------------
Pure-standard-library first build for validating the workflow.

What this version proves:
  1. the YouTube API key is loaded correctly
  2. candidate videos can be fetched from YouTube
  3. transcripts can be retrieved when available
  4. chunks can be ranked in a structured way

This is intentionally lightweight so you can run it immediately without
installing any extra packages in the current environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_TIMEDTEXT_URL = "https://video.google.com/timedtext"

_ENV_LOADED = False

DEMO_VIDEOS: list[dict[str, str]] = [
    {
        "video_id": "demo-sentiment-001",
        "title": "Build a Sentiment Analysis App End to End",
        "description": "Demo transcript for testing semantic ranking.",
        "channel": "Demo Channel",
        "url": "https://example.com/demo-sentiment-001",
        "transcript": (
            "In this tutorial we build a sentiment analysis project using Python. "
            "We clean text, train a classifier, and then deploy the model with a simple web interface."
        ),
    },
    {
        "video_id": "demo-flask-002",
        "title": "Deploy a Python Model with Flask",
        "description": "Demo transcript about deployment workflow.",
        "channel": "Demo Channel",
        "url": "https://example.com/demo-flask-002",
        "transcript": (
            "This video shows how to package a machine learning model and expose it with Flask. "
            "We discuss APIs, routes, prediction endpoints, and practical deployment steps."
        ),
    },
    {
        "video_id": "demo-nlp-003",
        "title": "NLP Project Ideas for Students",
        "description": "Demo transcript about project discovery.",
        "channel": "Demo Channel",
        "url": "https://example.com/demo-nlp-003",
        "transcript": (
            "Here we explore NLP project ideas such as sentiment analysis, topic classification, "
            "summarization, and retrieval systems for learning and experimentation."
        ),
    },
]


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    _ENV_LOADED = True


def get_api_key() -> str:
    _load_env_file()
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is missing from the environment or .env file.")
    return key


def api_key_status() -> dict[str, Any]:
    _load_env_file()
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    return {
        "present": bool(key),
        "length": len(key),
        "configured": bool(key),
    }


def expand_query(user_query: str) -> list[str]:
    base = user_query.strip()
    if not base:
        return []

    return [
        f"{base} tutorial",
        f"{base} python step by step",
        f"end to end {base} project",
    ]


def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(full_url, timeout=30) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network call failed for {url}: {exc}") from exc


def search_youtube(query: str, max_results: int = 10) -> list[dict[str, str]]:
    response = _http_get_json(
        YOUTUBE_SEARCH_URL,
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "relevanceLanguage": "en",
            "videoDuration": "medium",
            "key": get_api_key(),
        },
    )

    videos: list[dict[str, str]] = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        snippet = item.get("snippet", {})
        videos.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel": snippet.get("channelTitle", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )
    return videos


def fetch_candidates(user_query: str, results_per_phrase: int = 7) -> list[dict[str, str]]:
    phrases = expand_query(user_query)
    seen_ids: set[str] = set()
    all_videos: list[dict[str, str]] = []

    for phrase in phrases:
        results = search_youtube(phrase, max_results=results_per_phrase)
        for video in results:
            if video["video_id"] in seen_ids:
                continue
            seen_ids.add(video["video_id"])
            all_videos.append(video)

    return all_videos


def _decode_xml_text(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )


def get_transcript(video_id: str) -> str | None:
    params = {
        "v": video_id,
        "lang": "en",
    }
    query = urllib.parse.urlencode(params)
    full_url = f"{YOUTUBE_TIMEDTEXT_URL}?{query}"

    try:
        with urllib.request.urlopen(full_url, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="ignore")

        if not raw.strip():
            return None

        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return None

        texts: list[str] = []
        for node in root.findall(".//text"):
            text_value = node.text or ""
            text_value = _decode_xml_text(text_value.replace("\n", " ").strip())
            if text_value:
                texts.append(text_value)

        if not texts:
            return None

        return " ".join(texts)
    except Exception:
        return None


def chunk_text(text: str, chunk_size: int = 200, stride: int = 150) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += stride
        if start >= len(words):
            break
    return chunks


def collect_chunks(videos: list[dict[str, str]]) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for video in videos:
        transcript = get_transcript(video["video_id"])
        if not transcript:
            continue

        for chunk_idx, chunk_text_content in enumerate(chunk_text(transcript)):
            all_chunks.append(
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "url": video["url"],
                    "channel": video["channel"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text_content,
                }
            )
    return all_chunks


def _demo_candidates() -> list[dict[str, str]]:
    return [
        {
            "video_id": item["video_id"],
            "title": item["title"],
            "description": item["description"],
            "channel": item["channel"],
            "url": item["url"],
        }
        for item in DEMO_VIDEOS
    ]


def _demo_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for item in DEMO_VIDEOS:
        for chunk_idx, chunk_text_content in enumerate(chunk_text(item["transcript"])):
            chunks.append(
                {
                    "video_id": item["video_id"],
                    "title": item["title"],
                    "url": item["url"],
                    "channel": item["channel"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text_content,
                }
            )
    return chunks


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    shared = set(counter_a) & set(counter_b)
    dot = sum(counter_a[token] * counter_b[token] for token in shared)
    norm_a = math.sqrt(sum(value * value for value in counter_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counter_b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def score_chunk(query: str, chunk_text: str) -> float:
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk_text)
    base = cosine_similarity(query_tokens, chunk_tokens)

    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = len(query_set & chunk_set)
    coverage = overlap / max(1, len(query_set))
    phrase_bonus = 0.1 * coverage

    return min(1.0, base + phrase_bonus)


def rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {}

    for chunk in chunks:
        score = score_chunk(query, chunk["text"])
        vid = chunk["video_id"]
        grouped[vid].append((score, chunk))
        if vid not in meta:
            meta[vid] = {
                "video_id": vid,
                "title": chunk["title"],
                "url": chunk["url"],
                "channel": chunk["channel"],
            }

    results: list[dict[str, Any]] = []
    for vid, scored_chunks in grouped.items():
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_scores = [score for score, _ in scored_chunks]
        best_score = top_scores[0]
        top3_avg = sum(top_scores[:3]) / min(3, len(top_scores))
        final_score = 0.7 * best_score + 0.3 * top3_avg
        best_chunk = scored_chunks[0][1]

        results.append(
            {
                **meta[vid],
                "final_score": round(final_score, 4),
                "best_chunk_score": round(best_score, 4),
                "matched_snippet": best_chunk["text"][:300],
                "chunk_idx": best_chunk["chunk_idx"],
            }
        )

    results.sort(key=lambda item: item["final_score"], reverse=True)
    return results[:top_n]


@lru_cache(maxsize=32)
def run_search(
    query: str,
    top_n: int = 5,
    results_per_phrase: int = 7,
) -> dict[str, Any]:
    mode = "live"
    warnings: list[str] = []

    try:
        candidates = fetch_candidates(query, results_per_phrase=results_per_phrase)
        chunks = collect_chunks(candidates)
        if not chunks:
            warnings.append("No live transcripts were available, so the demo corpus was used.")
            mode = "offline-demo"
            candidates = _demo_candidates()
            chunks = _demo_chunks()
    except Exception as exc:
        warnings.append(str(exc))
        warnings.append("Live retrieval is unavailable, so the demo corpus was used.")
        mode = "offline-demo"
        candidates = _demo_candidates()
        chunks = _demo_chunks()

    results = rank_chunks(query, chunks, top_n=top_n)

    return {
        "query": query,
        "api_key": api_key_status(),
        "mode": mode,
        "warnings": warnings,
        "expanded_queries": expand_query(query),
        "total_candidates": len(candidates),
        "total_chunks": len(chunks),
        "results": results,
    }
