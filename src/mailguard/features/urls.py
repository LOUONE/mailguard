"""URL features: what the links of a message look like. Nothing is resolved or fetched.

URLs are present in 93.8% of phishing and 87.3% of ham, in every era, which makes them one of
the few signals that do not separate the corpora by date. Deliberately absent: the share of
`https` links, which mostly tells 2002 from 2020.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final
from urllib.parse import SplitResult, urlsplit

from mailguard.features.base import EmailFeatures
from mailguard.features.lists import SHORTENERS, SUSPICIOUS_TLDS, URL_KEYWORDS
from mailguard.features.psl import registrable_domain
from mailguard.parsing import Email
from mailguard.parsing.html import Anchor, read_html

URL_FEATURES: Final = (
    "url_count",
    "url_distinct_domains",
    "url_ip_hosts",
    "url_max_length",
    "url_max_host_entropy",
    "url_max_subdomain_depth",
    "url_punycode_hosts",
    "url_userinfo",
    "url_shorteners",
    "url_suspicious_tlds",
    "url_nonstandard_ports",
    "url_keyword_hits",
    "url_anchor_mismatches",
)

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# "mailto:x", "javascript:..." have no host; "example.org:8080" is a host with a port.
_HOSTLESS_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:(?!//|\d)", re.IGNORECASE)
# A domain written as link text: "www.bank.com", "https://bank.com/login", "Bank.com".
_DOMAIN_IN_TEXT_RE = re.compile(r"(?:[a-z0-9-]+\.)+[a-z]{2,63}", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedUrl:
    url: str
    host: str  # lowercased, may be empty
    domain: str  # registrable domain, the IP itself, or the host when no suffix matches
    is_ip: bool
    subdomain_depth: int
    port_is_standard: bool
    has_userinfo: bool


def is_ip_host(host: str) -> bool:
    """Dotted or IPv6 literal, or the integer and hex forms browsers also accept (`0x7f000001`)."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host.isdigit() or host.startswith("0x")
    return True


def parse_url(url: str) -> ParsedUrl | None:
    """Split a URL as a browser would read it. None when there is no usable host."""
    candidate = url.strip()
    if _HOSTLESS_SCHEME_RE.match(candidate):
        return None
    if not _SCHEME_RE.match(candidate):
        candidate = "http://" + candidate  # "www.example.org/x" as found in text
    try:
        parts: SplitResult = urlsplit(candidate)
        host = (parts.hostname or "").rstrip(".")
    except ValueError:  # "http://[::1" and other broken netlocs
        return None
    if not host:
        return None
    try:
        port_is_standard = parts.port in (None, 80, 443)
    except ValueError:  # port out of range or not a number
        port_is_standard = False

    is_ip = is_ip_host(host)
    domain = host if is_ip else registrable_domain(host) or host
    depth = 0 if is_ip else max(host.count(".") - domain.count("."), 0)
    return ParsedUrl(
        url=url,
        host=host,
        domain=domain,
        is_ip=is_ip,
        subdomain_depth=depth,
        port_is_standard=port_is_standard,
        has_userinfo="@" in parts.netloc,
    )


def shannon_entropy(text: str) -> float:
    """Bits per character. Random-looking hosts (`x7f9q2kz.example`) score high."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def anchor_mismatch(anchor: Anchor) -> bool:
    """The link text names a domain other than the one the link goes to."""
    shown = _DOMAIN_IN_TEXT_RE.search(anchor.text)
    if shown is None:
        return False
    shown_url, target = parse_url(shown.group()), parse_url(anchor.href)
    if shown_url is None or target is None:
        return False
    return shown_url.domain != target.domain


def url_features(urls: Iterable[str], anchors: Iterable[Anchor] = ()) -> dict[str, float]:
    parsed = [p for p in map(parse_url, urls) if p is not None]
    lengths = [len(p.url) for p in parsed]
    return {
        "url_count": float(len(parsed)),
        "url_distinct_domains": float(len({p.domain for p in parsed})),
        "url_ip_hosts": float(sum(p.is_ip for p in parsed)),
        "url_max_length": float(max(lengths, default=0)),
        "url_max_host_entropy": max((shannon_entropy(p.host) for p in parsed), default=0.0),
        "url_max_subdomain_depth": float(max((p.subdomain_depth for p in parsed), default=0)),
        "url_punycode_hosts": float(sum("xn--" in p.host for p in parsed)),
        "url_userinfo": float(sum(p.has_userinfo for p in parsed)),
        "url_shorteners": float(sum(p.domain in SHORTENERS for p in parsed)),
        "url_suspicious_tlds": float(
            sum(not p.is_ip and p.host.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for p in parsed)
        ),
        "url_nonstandard_ports": float(sum(not p.port_is_standard for p in parsed)),
        "url_keyword_hits": float(
            sum(any(k in p.url.casefold() for k in URL_KEYWORDS) for p in parsed)
        ),
        "url_anchor_mismatches": float(sum(map(anchor_mismatch, anchors))),
    }


class UrlFeatures(EmailFeatures):
    """URL features of each message, from `Email.urls` and the anchors of its HTML body."""

    names: ClassVar[tuple[str, ...]] = URL_FEATURES

    def extract(self, email: Email) -> Mapping[str, float]:
        anchors: Sequence[Anchor] = read_html(email.html_body).anchors if email.html_body else ()
        return url_features(email.urls, anchors)
