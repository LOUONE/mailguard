"""URL extraction from plain text and HTML. Nothing here resolves or fetches a URL."""

from __future__ import annotations

import re
from collections.abc import Iterable
from html.parser import HTMLParser

_URL_RE = re.compile(r"""(?:https?://|www\.)[^\s<>"'`]+""", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?)]}'\""
_LINK_ATTRS = {"a": "href", "area": "href", "form": "action"}


def _clean(url: str) -> str:
    return url.strip().rstrip(_TRAILING_PUNCT)


def urls_in_text(text: str) -> list[str]:
    """`http(s)://` and `www.` URLs found in free text, in order of appearance."""
    return [cleaned for m in _URL_RE.finditer(text) if (cleaned := _clean(m.group()))]


class _LinkCollector(HTMLParser):
    """Collects link targets and visible text, with HTML entities already decoded."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        wanted = _LINK_ATTRS.get(tag)
        for name, value in attrs:
            if name == wanted and value and _URL_RE.match(value.strip()):
                self.urls.append(_clean(value))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def urls_in_html(html: str) -> list[str]:
    """URLs in link targets (`<a href>`, `<area href>`, `<form action>`) and in visible text."""
    collector = _LinkCollector()
    try:
        collector.feed(html)
        collector.close()
    except AssertionError:  # html.parser still asserts on some malformed declarations
        pass
    return collector.urls + urls_in_text(" ".join(collector.text))


def dedupe(urls: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicates, keeping the first occurrence."""
    return tuple(dict.fromkeys(urls))
