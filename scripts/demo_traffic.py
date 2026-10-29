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

NAMES = ("alice", "bob", "carol", "dave", "erin", "frank")
TOPICS = (
    ("Lunch on Thursday?", "Are we still on for lunch on Thursday? I can bring the slides."),
    ("Re: quarterly report", "Thanks, the figures look right. I left two comments in section 3."),
    ("Your order has shipped", "Your parcel is on its way and should arrive within 3 days."),
    ("Team offsite", "The offsite moves to the 14th. Please confirm your dietary needs."),
    ("Weekly newsletter", "This week: five tips for better meetings, and our new podcast."),
)
LURES = (
    ("Urgent: verify your account", "Unusual activity was detected. Confirm your password at"),
    ("Your mailbox is full", "You have 5 pending messages. Upgrade your quota now at"),
    ("Payment declined", "We could not process your last payment. Update your billing at"),
    ("Invoice #{n} attached", "Please review the attached invoice and sign in to view it at"),
)
BRANDS = ("PayPal", "Microsoft", "DHL", "Netflix", "Your Bank")


def legitimate(rng: random.Random) -> str:
    sender, recipient = rng.sample(NAMES, 2)
    subject, body = rng.choice(TOPICS)
    return (
        f"Return-Path: <{sender}@example.org>\nFrom: {sender.title()} <{sender}@example.org>\n"
        f"To: {recipient}@example.org\nSubject: {subject}\n"
        f"Message-ID: <{rng.getrandbits(48):x}@example.org>\n"
        f"Date: Thu, 29 Oct 2026 10:00:00 +0000\n\nHi {recipient.title()},\n\n{body}\n\n"
        f"{sender.title()}\n"
    )


def phishing(rng: random.Random) -> str:
    brand = rng.choice(BRANDS)
    subject, lure = rng.choice(LURES)
    host = f"198.51.100.{rng.randint(1, 254)}"
    return (
        f'From: "{brand} Security" <no-reply@secure-{brand.lower().replace(" ", "")}.example>\n'
        f"Reply-To: support@other.example\nSubject: {subject.format(n=rng.randint(1000, 9999))}\n"
        f"Content-Type: text/html\n\n<p>Dear customer,</p><p>{lure} "
        f'<a href="http://{host}/login.php?id={rng.getrandbits(32):x}">www.{brand.lower()}.com</a>'
        "</p>\n"
    )


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
