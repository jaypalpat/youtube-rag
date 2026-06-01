"""
main1.py
--------
Standard-library HTTP server for the v1 build.

Run:
  python main1.py

Endpoints:
  GET  /               health check
  GET  /diagnose       structured end-to-end test
  GET  /search         semantic-lite search
  POST /search         same as GET but accepts JSON body
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pipeline1 import api_key_status, run_search

HOST = "127.0.0.1"
PORT = 8000


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "YouTubeSemanticSearchV1/0.1"

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _handle_search(self, query: str, top_n: int, results_per_phrase: int) -> None:
        if not query.strip():
            self._send_json(400, {"error": "Query cannot be empty."})
            return

        try:
            payload = run_search(
                query=query.strip(),
                top_n=top_n,
                results_per_phrase=results_per_phrase,
            )
            self._send_json(200, payload)
        except Exception as exc:
            print("[main1] search error")
            print(traceback.format_exc())
            self._send_json(500, {"error": str(exc)})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_json(
                200,
                {
                    "status": "running",
                    "message": "YouTube Semantic Search v1 is ready.",
                    "api_key": api_key_status(),
                    "usage": {
                        "diagnose": "/diagnose?query=sentiment+analysis+project",
                        "search": "/search?query=sentiment+analysis+project&top_n=5",
                    },
                },
            )
            return

        if parsed.path == "/diagnose":
            query = query_params.get("query", ["sentiment analysis project"])[0]
            top_n = int(query_params.get("top_n", ["5"])[0])
            self._handle_search(query=query, top_n=top_n, results_per_phrase=7)
            return

        if parsed.path == "/search":
            query = query_params.get("query", [""])[0]
            top_n = int(query_params.get("top_n", ["5"])[0])
            results_per_phrase = int(query_params.get("results_per_phrase", ["7"])[0])
            self._handle_search(
                query=query,
                top_n=top_n,
                results_per_phrase=results_per_phrase,
            )
            return

        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            body = self._read_json_body()
            query = str(body.get("query", ""))
            top_n = int(body.get("top_n", 5))
            results_per_phrase = int(body.get("results_per_phrase", 7))
            self._handle_search(
                query=query,
                top_n=top_n,
                results_per_phrase=results_per_phrase,
            )
        except Exception as exc:
            print("[main1] request error")
            print(traceback.format_exc())
            self._send_json(500, {"error": str(exc)})


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Serving on http://{HOST}:{PORT}")
    print("Try:")
    print("  /")
    print("  /diagnose?query=sentiment+analysis+project")
    print("  /search?query=sentiment+analysis+project&top_n=5")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
