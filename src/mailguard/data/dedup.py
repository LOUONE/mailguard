"""Content fingerprints used to find duplicate messages before splitting."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")
_NON_WORD_RE = re.compile(r"[\W_]+")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def normalise_text(text: str) -> str:
    """Reduce a body to what stays constant across copies of the same campaign.

    Unicode is NFKC-folded and lowercased, URLs and numbers are replaced by
    placeholders (they are personalised per recipient), and punctuation and
    whitespace are collapsed.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _URL_RE.sub(" url ", text)
    text = _DIGITS_RE.sub("0", text)
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


def exact_fingerprint(text: str) -> str | None:
    """Same body text, byte for byte once outer whitespace is stripped. None if empty."""
    stripped = text.strip()
    return _digest(stripped) if stripped else None


def normalised_fingerprint(text: str) -> str | None:
    """Same body text after `normalise_text`. None if nothing is left."""
    normalised = normalise_text(text)
    return _digest(normalised) if normalised else None
