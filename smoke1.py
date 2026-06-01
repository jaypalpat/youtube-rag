"""
smoke1.py
---------
CLI smoke test for the v1 build.

Usage:
  python smoke1.py "sentiment analysis flask deployment"
"""

from __future__ import annotations

import json
import sys

from pipeline1 import run_search


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or "sentiment analysis project"
    result = run_search(query=query, top_n=5, results_per_phrase=7)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
