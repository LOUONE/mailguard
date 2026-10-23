"""Minimal, dependency-free HTML reading: link targets and visible text."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

_LINK_ATTRS = {"a": "href", "area": "href", "form": "action"}
_INVISIBLE = frozenset({"script", "style", "head", "title", "template"})


@dataclass(frozen=True, slots=True)
class HtmlContent:
    links: tuple[str, ...]
    text: str


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.chunks: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INVISIBLE:
            self._hidden_depth += 1
        wanted = _LINK_ATTRS.get(tag)
        self.links.extend(v.strip() for k, v in attrs if k == wanted and v and v.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.chunks.append(data)


def read_html(html: str) -> HtmlContent:
    """Link targets (`a`, `area`, `form action`) and visible text, entities decoded."""
    collector = _Collector()
    try:
        collector.feed(html)
        collector.close()
    except AssertionError:  # html.parser has asserted on malformed declarations in the past
        pass
    return HtmlContent(
        links=tuple(collector.links),
        text=" ".join(" ".join(collector.chunks).split()),
    )
