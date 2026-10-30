"""Send a steady stream of made-up emails to the API, so the dashboard has something to show.

    python scripts/demo_traffic.py --url http://127.0.0.1:8000 --rate 2

Every message is synthetic (RFC 2606 domains, documentation IP ranges); about one in five is a
phishing lure. A few requests are deliberately invalid, so error rates are not flat zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

from mailguard.data.synthetic import personal as legitimate
from mailguard.data.synthetic import phishing


def request(rng: random.Random) -> dict[str, object]:
    roll = rng.random()
    if roll < 0.02:
        return {"raw": "", "unexpected": True}  # invalid on purpose: a 422
    return {"raw": phishing(rng) if roll < 0.22 else legitimate(rng)}


def post(url: str, body: dict[str, object]) -> tuple[int, str]:
    """POST JSON with the standard library only (the image has no HTTP client package)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response).get("label", "-")
    except urllib.error.HTTPError as exc:
        return exc.code, "-"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo_traffic")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=2.0, help="requests per second")
    parser.add_argument("--count", type=int, default=0, help="stop after this many (0: never)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    sent = 0
    while args.count == 0 or sent < args.count:
        try:
            status, label = post(f"{args.url}/v1/predict", request(rng))
            print(status, label, flush=True)
        except (urllib.error.URLError, OSError) as exc:
            print(f"api unreachable ({type(exc).__name__}), retrying", flush=True)
            time.sleep(2)
        sent += 1
        time.sleep(rng.expovariate(args.rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
