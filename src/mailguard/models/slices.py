"""Slices of an evaluation set where the model is expected to be weak, from the error analysis.

The real misses of v0.1.0 share one pattern: little English text for the model to read (other
languages, junk padding, lures in an attachment or an image). These slices turn that observation
into numbers the report tracks. The bin edges were fixed on the training split, never on test.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from mailguard.features.text import scrubbed_text
from mailguard.parsing import Email

# The most frequent English function words. They make up about 40% of English prose and almost
# none of German, Japanese or Chinese text, which makes their share a crude but dependency-free
# language signal.
ENGLISH_FUNCTION_WORDS: Final = frozenset(
    {
        "the", "and", "to", "of", "a", "in", "you", "your", "is", "for", "that",
        "this", "it", "on", "with", "be", "are", "we", "will", "as", "or", "at",
        "by", "from", "have", "not", "please", "our", "if", "can", "an", "all",
    }
)  # fmt: skip
# What `scrubbed_text` puts in place of addresses and URLs: not words the sender wrote.
_PLACEHOLDERS: Final = frozenset({"emailaddress", "urladdress"})
_WORD_RE = re.compile(r"[^\W\d_]+")

SHORT_TEXT: Final = 20  # words
LONG_TEXT: Final = 100
MIN_ENGLISH_SHARE: Final = 0.1


def readable_words(email: Email) -> list[str]:
    """Words of the text the model reads, lowercased, without placeholders or digits."""
    words = _WORD_RE.findall(scrubbed_text(email).casefold())
    return [w for w in words if w not in _PLACEHOLDERS]


def english_share(words: Sequence[str]) -> float:
    """Share of `words` that are common English function words; 0 for no words."""
    return sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words) if words else 0.0


def slice_masks(emails: Sequence[Email]) -> dict[str, NDArray[np.bool_]]:
    """One boolean mask per slice, in report order. Language is only judged on 20+ words."""
    words = [readable_words(email) for email in emails]
    counts = np.array([len(w) for w in words], dtype=np.int64)
    english = np.array([english_share(w) >= MIN_ENGLISH_SHARE for w in words], dtype=bool)
    enough = counts >= SHORT_TEXT
    return {
        f"under {SHORT_TEXT} words": ~enough,
        f"{SHORT_TEXT} to {LONG_TEXT - 1} words": enough & (counts < LONG_TEXT),
        f"{LONG_TEXT} words or more": counts >= LONG_TEXT,
        f"{SHORT_TEXT}+ words, mostly English": enough & english,
        f"{SHORT_TEXT}+ words, little English": enough & ~english,
    }
