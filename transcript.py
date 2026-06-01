"""
transcript.py
-------------
Downloads transcripts (captions) for YouTube videos and splits them into
overlapping chunks that are ready for embedding.

Fetches captions via youtube-transcript-api first, then yt-dlp as fallback.
Requests are throttled to reduce YouTube 429 rate-limit errors.
Successful transcripts are cached on disk by video_id to avoid repeat calls.
"""

from __future__ import annotations

import os
import random
import re
import tempfile
import time
from pathlib import Path

import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

_ytt_api = YouTubeTranscriptApi()
_REQUEST_DELAY_SEC = float(os.getenv("YT_REQUEST_DELAY_SEC", "1.0"))
_MAX_REQUEST_DELAY_SEC = float(os.getenv("YT_MAX_REQUEST_DELAY_SEC", "16.0"))
_RATE_LIMIT_COOLDOWN_SEC = float(os.getenv("YT_RATE_LIMIT_COOLDOWN_SEC", "30.0"))
_MAX_TRANSCRIPT_ATTEMPTS = int(os.getenv("YT_TRANSCRIPT_ATTEMPTS", "3"))
_last_request_at = 0.0
_backoff_sec = _REQUEST_DELAY_SEC
_cooldown_until = 0.0

_TRANSCRIPT_CACHE_DIR = Path(
    os.getenv(
        "YT_TRANSCRIPT_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".cache" / "transcripts"),
    )
)

_YDL_SUB_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en", "en-US", "en-GB"],
    "subtitlesformat": "vtt/best",
    "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    "retries": 2,
    "sleep_interval_requests": 0.5,
}


def _rate_limit() -> None:
    global _last_request_at
    now = time.time()
    cooldown_wait = _cooldown_until - now
    pacing_wait = _backoff_sec - (now - _last_request_at)
    wait = max(cooldown_wait, pacing_wait, 0.0)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _register_rate_limit() -> None:
    global _backoff_sec, _cooldown_until
    _backoff_sec = min(_MAX_REQUEST_DELAY_SEC, max(_REQUEST_DELAY_SEC, _backoff_sec * 2))
    jitter = random.uniform(0, _backoff_sec * 0.35)
    cooldown = _RATE_LIMIT_COOLDOWN_SEC + _backoff_sec + jitter
    _cooldown_until = max(_cooldown_until, time.time() + cooldown)


def _register_success() -> None:
    global _backoff_sec, _cooldown_until
    _backoff_sec = _REQUEST_DELAY_SEC
    _cooldown_until = 0.0


def _cache_path(video_id: str) -> Path:
    safe_video_id = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)
    return _TRANSCRIPT_CACHE_DIR / f"{safe_video_id}.txt"


def _load_cached_transcript(video_id: str) -> str | None:
    path = _cache_path(video_id)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError as e:
        print(f"  [transcript] cache read failed ({video_id}): {e}")
        return None
    if text:
        print(f"  [transcript] {video_id} → cache hit")
        return text
    return None


def _save_cached_transcript(video_id: str, text: str) -> None:
    content = text.strip()
    if not content:
        return
    try:
        _TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(video_id).write_text(content, encoding="utf-8")
    except OSError as e:
        print(f"  [transcript] cache write failed ({video_id}): {e}")


def _parse_vtt(vtt_text: str) -> str:
    lines: list[str] = []
    for raw in vtt_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def _get_transcript_ytt_api(video_id: str) -> str | None:
    transcript_list = _ytt_api.list(video_id)
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"]).fetch()
    except Exception:
        transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"]).fetch()

    full_text = " ".join(
        snippet.text.replace("\n", " ").strip()
        for snippet in transcript
    )
    return full_text or None


def _get_transcript_ytdlp(video_id: str) -> str | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            **_YDL_SUB_OPTS,
            "outtmpl": os.path.join(tmpdir, "%(id)s"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for name in os.listdir(tmpdir):
            if not name.endswith(".vtt"):
                continue
            path = os.path.join(tmpdir, name)
            text = _parse_vtt(open(path, encoding="utf-8", errors="ignore").read())
            if text.strip():
                return text
    return None


def _is_rate_limit_error(err: str) -> bool:
    lowered = err.lower()
    return any(token in lowered for token in ("429", "block", "too many requests", "ipblocked"))


def get_transcript(video_id: str) -> str | None:
    """
    Download transcript text for a video, trying multiple methods with retries.
    """
    cached = _load_cached_transcript(video_id)
    if cached:
        return cached

    now = time.time()
    if now < _cooldown_until:
        wait = _cooldown_until - now
        print(f"  [transcript] cooling down ({wait:.1f}s) before new caption attempts")
        return None

    for attempt in range(_MAX_TRANSCRIPT_ATTEMPTS):
        _rate_limit()

        try:
            text = _get_transcript_ytt_api(video_id)
            if text:
                _register_success()
                _save_cached_transcript(video_id, text)
                return text
        except (NoTranscriptFound, TranscriptsDisabled):
            break
        except Exception as e:
            err = str(e)
            if _is_rate_limit_error(err):
                _register_rate_limit()
                print(
                    "  [transcript] Caption endpoint rate-limited on ytt-api; "
                    f"backoff now {_backoff_sec:.1f}s"
                )
                break
            print(f"  [transcript] ytt-api {video_id}: {err[:120]}")

        _rate_limit()
        try:
            text = _get_transcript_ytdlp(video_id)
            if text:
                _register_success()
                _save_cached_transcript(video_id, text)
                return text
        except Exception as e:
            err = str(e)
            if _is_rate_limit_error(err):
                _register_rate_limit()
                print(
                    "  [transcript] Caption endpoint rate-limited on yt-dlp; "
                    f"backoff now {_backoff_sec:.1f}s"
                )
                break
            print(f"  [transcript] yt-dlp {video_id}: {err[:120]}")

        if attempt < _MAX_TRANSCRIPT_ATTEMPTS - 1:
            retry_wait = min(_MAX_REQUEST_DELAY_SEC, _backoff_sec) + random.uniform(0.0, 0.5)
            time.sleep(retry_wait)

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


def metadata_chunks_for_video(video: dict) -> list[dict]:
    """Fallback when captions are unavailable: rank using title + description."""
    title = (video.get("title") or "").strip()
    description = (video.get("description") or "").strip()
    channel = (video.get("channel") or "").strip()
    text = ". ".join(part for part in (title, description, channel) if part)
    if not text:
        return []

    return [
        {
            "video_id": video["video_id"],
            "title": title,
            "url": video["url"],
            "channel": channel,
            "chunk_idx": 0,
            "text": text,
            "source": "metadata",
        }
    ]


def get_chunks_for_video(video: dict, allow_metadata_fallback: bool = True) -> list[dict] | None:
    video_id = video["video_id"]
    transcript = get_transcript(video_id)

    if transcript:
        result = []
        for idx, chunk_text_content in enumerate(chunk_text(transcript)):
            result.append(
                {
                    "video_id": video_id,
                    "title": video["title"],
                    "url": video["url"],
                    "channel": video["channel"],
                    "chunk_idx": idx,
                    "text": chunk_text_content,
                    "source": "transcript",
                }
            )
        print(f"  [transcript] {video['title'][:50]} → {len(result)} chunks")
        return result

    if allow_metadata_fallback:
        meta = metadata_chunks_for_video(video)
        if meta:
            print(f"  [transcript] {video['title'][:50]} → metadata fallback")
            return meta

    print(f"  [transcript] No transcript for: {video['title'][:60]}")
    return None


def process_all_videos(
    videos: list[dict],
    min_chunks: int = 1,
    max_videos_to_probe: int = 8,
    allow_metadata_fallback: bool = True,
) -> tuple[list[dict], bool]:
    """
    Collect chunks from candidate videos.

    Returns (chunks, used_metadata_fallback).
    """
    all_chunks: list[dict] = []
    used_metadata = False
    transcript_chunks = 0
    limit = min(len(videos), max_videos_to_probe)
    print(f"[transcript] Processing up to {limit}/{len(videos)} videos...\n")

    for video in videos[:limit]:
        chunks = get_chunks_for_video(video, allow_metadata_fallback=allow_metadata_fallback)
        if not chunks:
            continue

        if chunks[0].get("source") == "metadata":
            used_metadata = True
        else:
            transcript_chunks += len(chunks)

        all_chunks.extend(chunks)

        if transcript_chunks >= min_chunks:
            break

    print(f"\n[transcript] Total chunks collected: {len(all_chunks)}")
    return all_chunks, used_metadata
