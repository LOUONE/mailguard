"""Minimal, dependency-free HTML reading: link targets, anchor texts and visible text."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

_LINK_ATTRS = {"a": "href", "area": "href", "form": "action"}
_INVISIBLE = frozenset({"script", "style", "head", "title", "template"})


@dataclass(frozen=True, slots=True)
class Anchor:
    """An `<a href>` and the visible text between its tags."""

    href: str
    text: str


@dataclass(frozen=True, slots=True)
class HtmlContent:
    links: tuple[str, ...]
    text: str
    anchors: tuple[Anchor, ...] = ()


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.chunks: list[str] = []
        self.anchors: list[Anchor] = []
        self._hidden_depth = 0
        self._anchor: tuple[str, list[str]] | None = None  # href and text of the open <a>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INVISIBLE:
            self._hidden_depth += 1
        wanted = _LINK_ATTRS.get(tag)
        targets = [v.strip() for k, v in attrs if k == wanted and v and v.strip()]
        self.links.extend(targets)
        if tag == "a":
            self._close_anchor()  # an unclosed <a> ends where the next one starts
            if targets:
                self._anchor = (targets[0], [])

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE and self._hidden_depth:
            self._hidden_depth -= 1
        if tag == "a":
            self._close_anchor()

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.chunks.append(data)
            if self._anchor is not None:
                self._anchor[1].append(data)

    def _close_anchor(self) -> None:
        if self._anchor is not None:
            href, chunks = self._anchor
            # Inline tags inside a link (`www.<b>bank</b>.com`) must not split the text.
            self.anchors.append(Anchor(href, " ".join("".join(chunks).split())))
            self._anchor = None


def read_html(html: str) -> HtmlContent:
    """Link targets (`a`, `area`, `form action`), anchors and visible text, entities decoded."""
    collector = _Collector()
    try:
        collector.feed(html)
        collector.close()
    except AssertionError:  # html.parser has asserted on malformed declarations in the past
        pass
    collector._close_anchor()  # an <a> left open at the end of the document
    return HtmlContent(
        links=tuple(collector.links),
        text=" ".join(" ".join(collector.chunks).split()),
        anchors=tuple(collector.anchors),
    )
