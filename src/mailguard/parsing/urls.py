"""URL extraction from plain text and HTML. Nothing here resolves or fetches a URL."""

from __future__ import annotations

import re
from collections.abc import Iterable

from mailguard.parsing.html import read_html

_URL_RE = re.compile(r"""(?:https?://|www\.)[^\s<>"'`]+""", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?)]}'\""


def _clean(url: str) -> str:
    return url.strip().rstrip(_TRAILING_PUNCT)


def urls_in_text(text: str) -> list[str]:
    """`http(s)://` and `www.` URLs found in free text, in order of appearance."""
    return [cleaned for m in _URL_RE.finditer(text) if (cleaned := _clean(m.group()))]


def urls_in_html(html: str) -> list[str]:
    """URLs in link targets (`<a href>`, `<area href>`, `<form action>`) and in visible text."""
    content = read_html(html)
    targets = [_clean(link) for link in content.links if _URL_RE.match(link)]
    return targets + urls_in_text(content.text)


def dedupe(urls: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicates, keeping the first occurrence."""
    return tuple(dict.fromkeys(urls))
