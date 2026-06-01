"""
search.py
---------
Handles two things:
  1. Query expansion  - turns your one query into 3 related search phrases
                        so we fish a wider net on YouTube before semantic ranking.
  2. YouTube search   - uses yt-dlp (no API key) to find candidate videos.

Why expand queries?
  YouTube's search is keyword-based. If you type "sentiment analysis NLP",
  you might miss a great video titled "build a text classifier with Flask".
  By generating a few related phrases, we collect more candidates for the
  semantic stage to rank properly.
"""

import html

import yt_dlp

_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
}


def expand_query(user_query: str) -> list[str]:
    """
    Turn one user query into 3 related YouTube search phrases.

    This is a simple rule-based expander for the MVP.
    In Phase 2 you can replace this with an LLM call for smarter expansion.
    """
    base = user_query.strip()
    expansions = [
        f"{base} tutorial",
        f"{base} python step by step",
        f"end to end {base} project",
    ]
    return expansions


def search_youtube(query: str, max_results: int = 10) -> list[dict]:
    """
    Search YouTube for a single query string via yt-dlp (free, no API key).

    Returns a list of video dicts: {video_id, title, description, channel}.
    """
    videos: list[dict] = []

    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)

    for entry in info.get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id")
        if not video_id:
            continue
        videos.append(
            {
                "video_id": video_id,
                "title": html.unescape(entry.get("title") or ""),
                "description": html.unescape(entry.get("description") or ""),
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )

    return videos


def fetch_candidates(user_query: str, results_per_phrase: int = 10) -> list[dict]:
    """
    Full candidate-fetch pipeline:
      1. Expand the user query into multiple phrases.
      2. Search YouTube for each phrase.
      3. Deduplicate by video_id (same video might show up in multiple searches).

    Returns a deduplicated list of candidate video dicts.
    """
    phrases = expand_query(user_query)
    print(f"\n[search] Expanded to {len(phrases)} phrases:")
    for p in phrases:
        print(f"  → {p}")

    seen_ids: set[str] = set()
    all_videos: list[dict] = []

    for phrase in phrases:
        results = search_youtube(phrase, max_results=results_per_phrase)
        for video in results:
            if video["video_id"] not in seen_ids:
                seen_ids.add(video["video_id"])
                all_videos.append(video)

    print(f"[search] Fetched {len(all_videos)} unique candidate videos.\n")
    return all_videos
