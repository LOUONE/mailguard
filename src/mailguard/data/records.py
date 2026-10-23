"""One privacy-safe summary row per message, for exploration and dataset checks.

A record holds counts, flags and hashes only: no text, no address, no URL.
Notebook outputs built from records can be committed without leaking the
content of the corpora (see docs/data.md, "Personal data").
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mailguard.data.corpus import RawMessage, iter_corpus
from mailguard.data.dedup import exact_fingerprint, normalised_fingerprint
from mailguard.data.sources import SOURCES, Corpus, Label, Source
from mailguard.parsing import parse_email


@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: str
    corpus: Corpus
    file: str
    label: Label
    raw_bytes: int
    n_headers: int
    n_received: int
    has_text: bool
    has_html: bool
    body_chars: int
    n_urls: int
    n_attachments: int
    n_defects: int
    date: datetime | None  # from the Date header, normalised to UTC; untrusted
    exact_fp: str | None
    normalised_fp: str | None


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    # A naive datetime comes from "-0000", i.e. "UTC, origin unknown" (RFC 5322).
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def to_record(message: RawMessage) -> MessageRecord:
    email = parse_email(message.raw)
    body = email.body_text
    return MessageRecord(
        id=message.id,
        corpus=message.source.corpus,
        file=message.source.filename,
        label=message.source.label,
        raw_bytes=len(message.raw),
        n_headers=len(email.headers),
        n_received=len(email.header_values("Received")),
        has_text=bool(email.text_body.strip()),
        has_html=bool(email.html_body.strip()),
        body_chars=len(body),
        n_urls=len(email.urls),
        n_attachments=len(email.attachments),
        n_defects=len(email.defects),
        date=_as_utc(email.date),
        exact_fp=exact_fingerprint(body),
        normalised_fp=normalised_fingerprint(body),
    )


def iter_records(dest: Path, sources: Iterable[Source] = SOURCES) -> Iterator[MessageRecord]:
    """Parse the corpora lazily, yielding one record per message."""
    return (to_record(message) for message in iter_corpus(dest, sources))
