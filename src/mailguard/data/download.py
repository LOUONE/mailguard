"""Download the corpora listed in `sources.SOURCES` and verify their SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, get_args
from urllib.request import Request, urlopen

from mailguard.config import get_settings
from mailguard.data.sources import SOURCES, Corpus, Source

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20  # 1 MiB
TIMEOUT_S = 60.0
USER_AGENT = "mailguard-downloader (+https://github.com/LOUONE/mailguard)"

Fetcher = Callable[[str], Iterator[bytes]]
Outcome = Literal["cached", "downloaded"]


class ChecksumError(Exception):
    """A downloaded file does not match its pinned SHA-256."""

    def __init__(self, source: Source, actual: str) -> None:
        super().__init__(
            f"checksum mismatch for {source.filename}: expected {source.sha256}, got {actual}. "
            f"The upstream file may have changed; check {source.url} before updating the manifest."
        )
        self.source = source
        self.actual = actual


def http_chunks(url: str) -> Iterator[bytes]:
    """Stream a URL in chunks so large archives never sit fully in memory."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=TIMEOUT_S) as response:
        while chunk := response.read(CHUNK_SIZE):
            yield chunk


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def target_path(source: Source, dest: Path) -> Path:
    return dest / source.corpus / source.filename


def ensure_source(
    source: Source, dest: Path, *, fetch: Fetcher = http_chunks, force: bool = False
) -> Outcome:
    """Make sure `source` is on disk with the right checksum, downloading it if needed."""
    target = target_path(source, dest)
    if target.exists() and not force:
        if sha256_file(target) == source.sha256:
            return "cached"
        logger.warning("%s: checksum mismatch on disk, downloading again", target)

    target.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename at the end, so an interrupted
    # download never leaves a truncated file under the final name.
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with partial.open("wb") as out:
            for chunk in fetch(source.url):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != source.sha256:
            raise ChecksumError(source, actual)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return "downloaded"


def download_all(
    sources: Iterable[Source], dest: Path, *, fetch: Fetcher = http_chunks, force: bool = False
) -> list[Source]:
    """Fetch every source, keep going on errors, and return the ones that failed."""
    failed: list[Source] = []
    for source in sources:
        try:
            outcome = ensure_source(source, dest, fetch=fetch, force=force)
        except (ChecksumError, OSError) as exc:  # URLError and timeouts are OSError
            logger.error("%s: %s", source.filename, exc)
            failed.append(source)
        else:
            logger.info("%-28s %s", source.filename, outcome)
    return failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Download the public corpora listed in docs/data.md and verify their SHA-256.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: $MAILGUARD_DATA_DIR/raw)",
    )
    parser.add_argument(
        "--corpus",
        choices=get_args(Corpus),
        default=None,
        help="only download this corpus (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if the file is already present and valid",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    sources: Sequence[Source] = SOURCES,
    fetch: Fetcher = http_chunks,
) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(message)s")

    dest: Path = args.dest or settings.data_dir / "raw"
    selected = [s for s in sources if args.corpus in (None, s.corpus)]
    failed = download_all(selected, dest, fetch=fetch, force=args.force)

    if failed:
        logger.error("%d of %d files failed", len(failed), len(selected))
        return 1
    logger.info("%d files ready in %s", len(selected), dest)
    return 0
