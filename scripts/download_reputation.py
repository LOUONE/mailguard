"""Download the phishing feeds the agent's `lookup_reputation` tool searches (ADR 0010).

    uv run python scripts/download_reputation.py

Writes each feed into `MAILGUARD_REPUTATION_DIR` (data/reputation/, gitignored) with a manifest of
SHA-256 checksums and download times. The feeds change every hour, so a checksum cannot be
pinned in the repository as for the corpora: it is recorded here and checked at every load, so a
file edited or truncated after download is refused instead of silently trusted.

The feeds are third-party data under their providers' terms (PhishTank, OpenPhish); they are
never committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime

import httpx

from mailguard.agent.tools.network import FEEDS, MANIFEST, ReputationIndex
from mailguard.config import get_settings

TIMEOUT_S = 120.0


def main() -> int:
    directory = get_settings().reputation_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT_S) as http:
        for feed in FEEDS:
            try:
                response = http.get(feed.url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"error: {feed.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            (directory / feed.file).write_bytes(response.content)
            manifest[feed.name] = {
                "url": feed.url,
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    index = ReputationIndex.load(directory)
    for info in index.feeds:
        print(f"{info.feed}: {info.entries} URLs")
    print(f"{len(index.by_url)} distinct URLs on {len(index.by_host)} hosts in {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
