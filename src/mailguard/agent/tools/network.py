"""Investigation tools that need data from outside: domain age and URL reputation (ADR 0010).

Neither tool ever touches what the email links to.

- `domain_age` asks the registry over RDAP (RFC 9083, JSON over HTTPS). The server comes from the
  IANA bootstrap file bundled with the package, and every request goes through
  `AllowlistTransport`, which refuses any host that is not one of those registry servers. Only
  domains that appear in the email can be looked up, so a message cannot point the agent at
  something else.
- `lookup_reputation` never goes online: it searches phishing feeds downloaded beforehand by
  `make reputation` (PhishTank verified-online, OpenPhish community), whose checksums are
  recorded at download and checked at load.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import html
import io
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from mailguard.agent.tools import Arguments, Case, Output, Tool, clip
from mailguard.features.headers import domain_of
from mailguard.features.lists import SHORTENERS
from mailguard.features.psl import registrable_domain
from mailguard.features.urls import parse_url

BOOTSTRAP_FILE: Final = "rdap_dns.json"
RDAP_CACHE_S: Final = 24 * 3600
MANIFEST: Final = "manifest.json"
# A host carrying this many listed URLs is a platform many campaigns abuse, not a campaign's own
# host. On the 2026-11-03 snapshot (22,914 hosts): median 1 listed URL per host, 99th percentile
# 4, and 34 hosts with 50 or more, led by qrco.de (4,592), docs.google.com (4,176), bit.ly.
SHARED_HOST_LISTINGS: Final = 50

# --- the network fence -----------------------------------------------------------------------


class BlockedHost(httpx.TransportError):
    """A request to a host outside the allowlist. Raised before any connection is made."""


class AllowlistTransport(httpx.AsyncBaseTransport):
    """Lets HTTPS requests through to the listed hosts only."""

    def __init__(
        self, allowed_hosts: frozenset[str], inner: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._allowed = allowed_hosts
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != "https" or request.url.host not in self._allowed:
            raise BlockedHost(f"{request.url.scheme}://{request.url.host} is not allowed")
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# --- RDAP ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RdapBootstrap:
    """IANA's map from TLD to RDAP server (https://data.iana.org/rdap/dns.json), a snapshot."""

    servers: Mapping[str, str]  # tld -> base URL, ending in "/"
    publication: str

    @classmethod
    def parse(cls, document: Mapping[str, Any]) -> RdapBootstrap:
        servers: dict[str, str] = {}
        for tlds, urls in document["services"]:
            https = [u for u in urls if u.startswith("https://")]
            if https:
                base = https[0] if https[0].endswith("/") else https[0] + "/"
                servers.update(dict.fromkeys((t.casefold() for t in tlds), base))
        return cls(servers, str(document.get("publication", "")))

    def server_for(self, domain: str) -> str | None:
        return self.servers.get(domain.rsplit(".", 1)[-1].casefold())

    @property
    def hosts(self) -> frozenset[str]:
        return frozenset(httpx.URL(base).host for base in self.servers.values())


@cache
def default_bootstrap() -> RdapBootstrap:
    text = resources.files("mailguard.agent.tools").joinpath("data", BOOTSTRAP_FILE).read_text()
    return RdapBootstrap.parse(json.loads(text))


RdapStatus = Literal["registered", "not_found", "no_rdap_service", "unavailable", "not_in_email"]


class DomainRecord(Output):
    domain: str
    status: RdapStatus
    registered_at: str | None = None
    age_days: int | None = Field(None, description="Days since registration.")
    expires_at: str | None = None
    registrar: str | None = None
    statuses: tuple[str, ...] = ()
    source: str | None = Field(None, description="The registry's RDAP server.")
    detail: str | None = None


def _event(document: Mapping[str, Any], action: str) -> str | None:
    for event in document.get("events") or ():
        if isinstance(event, dict) and event.get("eventAction") == action:
            date = event.get("eventDate")
            return str(date) if date else None
    return None


def _registrar(document: Mapping[str, Any]) -> str | None:
    for entity in document.get("entities") or ():
        if not isinstance(entity, dict) or "registrar" not in (entity.get("roles") or ()):
            continue
        vcard = entity.get("vcardArray")
        properties = vcard[1] if isinstance(vcard, list) and len(vcard) > 1 else ()
        for prop in properties:
            if isinstance(prop, list) and len(prop) >= 4 and prop[0] == "fn":
                return clip(str(prop[3]), 100)
    return None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class RdapClient:
    """Registration data for a domain, from its registry, with a one-day cache."""

    def __init__(
        self,
        bootstrap: RdapBootstrap,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bootstrap = bootstrap
        self._timeout_s = timeout_s
        self._now = now
        self._clock = clock
        self._cache: dict[str, tuple[float, DomainRecord]] = {}
        self._http = httpx.AsyncClient(
            transport=AllowlistTransport(bootstrap.hosts, transport),
            timeout=timeout_s,
            follow_redirects=False,
            headers={"Accept": "application/rdap+json"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def lookup(self, domain: str) -> DomainRecord:
        cached = self._cache.get(domain)
        if cached and self._clock() - cached[0] < RDAP_CACHE_S:
            return cached[1]
        record = await self._fetch(domain)
        if record.status in ("registered", "not_found"):
            self._cache[domain] = (self._clock(), record)
        return record

    async def _fetch(self, domain: str) -> DomainRecord:
        base = self._bootstrap.server_for(domain)
        if base is None:
            tld = domain.rsplit(".", 1)[-1]
            return DomainRecord(domain=domain, status="no_rdap_service", detail=f".{tld}")
        source = httpx.URL(base).host
        try:
            async with asyncio.timeout(self._timeout_s):
                response = await self._http.get(f"{base}domain/{domain}")
        except TimeoutError:
            return DomainRecord(
                domain=domain, status="unavailable", source=source, detail="timeout"
            )
        except httpx.HTTPError as exc:
            detail = type(exc).__name__
            return DomainRecord(domain=domain, status="unavailable", source=source, detail=detail)
        if response.status_code == 404:
            return DomainRecord(domain=domain, status="not_found", source=source)
        if response.status_code != 200:
            detail = f"HTTP {response.status_code}"
            return DomainRecord(domain=domain, status="unavailable", source=source, detail=detail)
        try:
            document = response.json()
        except ValueError:
            return DomainRecord(
                domain=domain, status="unavailable", source=source, detail="not JSON"
            )
        if not isinstance(document, dict):
            return DomainRecord(
                domain=domain, status="unavailable", source=source, detail="not JSON"
            )
        registered = _event(document, "registration")
        when = _parse_date(registered)
        return DomainRecord(
            domain=domain,
            status="registered",
            registered_at=registered,
            age_days=(self._now() - when).days if when else None,
            expires_at=_event(document, "expiration"),
            registrar=_registrar(document),
            statuses=tuple(str(s)[:60] for s in (document.get("status") or ())[:10]),
            source=source,
        )


def case_domains(case: Case) -> set[str]:
    """Registrable domains the email names: sender addresses and links."""
    email = case.email
    found = {domain_of(a) for a in (email.from_address, email.reply_to, email.return_path)}
    found |= {p.domain for u in email.urls if (p := parse_url(u)) is not None and not p.is_ip}
    return {d for d in found if d}


class DomainArguments(Arguments):
    domain: str = Field(
        min_length=3,
        max_length=253,
        pattern=r"^[A-Za-z0-9.-]+$",
        description="A domain from the email (sender or link), e.g. example.org.",
    )


def domain_age_tool(rdap: RdapClient) -> Tool[DomainArguments, DomainRecord]:
    async def domain_age(case: Case, arguments: DomainArguments) -> DomainRecord:
        asked = arguments.domain.casefold().strip(".")
        domain = registrable_domain(asked) or asked
        if domain not in case_domains(case):
            return DomainRecord(domain=domain, status="not_in_email")
        return await rdap.lookup(domain)

    return Tool(
        name="domain_age",
        description=(
            "When a domain from the email was registered, by whom, and its registry status, "
            "from the registry's RDAP server. Phishing domains are often days or weeks old. "
            "Only domains that appear in the email (sender, reply-to, links) can be looked up. "
            "`no_rdap_service` and `unavailable` mean unknown, not suspicious: many country "
            "domains (.de, .ch) have no RDAP server."
        ),
        arguments=DomainArguments,
        run=domain_age,
    )


# --- reputation ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Feed:
    name: str
    url: str
    file: str
    fmt: Literal["phishtank", "lines"]


FEEDS: Final = (
    Feed(
        "phishtank",
        "https://data.phishtank.com/data/online-valid.csv",
        "phishtank.csv",
        "phishtank",
    ),
    Feed("openphish", "https://openphish.com/feed.txt", "openphish.txt", "lines"),
)


@dataclass(frozen=True, slots=True)
class Listing:
    feed: str
    target: str | None = None
    added: str | None = None


def url_key(url: str) -> str | None:
    """A URL as feeds and emails can both be compared on: no scheme, no fragment, no `&amp;`."""
    text = html.unescape(url.strip())
    parsed = parse_url(text)  # None for "mailto:", broken hosts and empty strings
    if parsed is None:
        return None
    parts = urlsplit(text if "://" in text else "http://" + text)
    netloc = parsed.host if parsed.port_is_standard else f"{parsed.host}:{parts.port}"
    path = parts.path.rstrip("/")
    return netloc + path + (f"?{parts.query}" if parts.query else "")


def _host_of(key: str) -> str:
    return key.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0]


def read_feed(feed: Feed, text: str) -> list[tuple[str, Listing]]:
    if feed.fmt == "lines":
        rows = [(line, Listing(feed.name)) for line in text.splitlines() if line.strip()]
    else:
        rows = [
            (r["url"], Listing(feed.name, r.get("target") or None, r.get("submission_time")))
            for r in csv.DictReader(io.StringIO(text))
            if r.get("url")
        ]
    return [(key, listing) for url, listing in rows if (key := url_key(url))]


class FeedInfo(Output):
    feed: str
    downloaded_at: str
    age_days: int
    entries: int


@dataclass
class ReputationIndex:
    by_url: dict[str, list[Listing]] = field(default_factory=lambda: defaultdict(list))
    by_host: Counter[str] = field(default_factory=Counter)
    feeds: list[FeedInfo] = field(default_factory=list)

    @classmethod
    def load(
        cls, directory: Path, now: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> ReputationIndex:
        """Read the downloaded feeds, refusing any file whose checksum changed since download."""
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        index = cls()
        for feed in FEEDS:
            entry = manifest.get(feed.name)
            if entry is None:
                continue
            data = (directory / feed.file).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"{feed.file} does not match its recorded checksum")
            pairs = read_feed(feed, data.decode("utf-8", errors="replace"))
            for key, listing in pairs:
                index.by_url[key].append(listing)
                index.by_host[_host_of(key)] += 1
            downloaded = _parse_date(entry["downloaded_at"]) or now()
            index.feeds.append(
                FeedInfo(
                    feed=feed.name,
                    downloaded_at=entry["downloaded_at"],
                    age_days=(now() - downloaded).days,
                    entries=len(pairs),
                )
            )
        return index


class ListingOut(Output):
    feed: str
    target: str | None
    added: str | None


class Reputation(Output):
    url: str
    available: bool = Field(description="False when no feed snapshot is loaded.")
    listed: bool = Field(description="This exact URL is in a phishing feed.")
    listings: tuple[ListingOut, ...] = ()
    same_host_listings: int = Field(0, description="Listed phishing URLs on the same host.")
    shared_platform: bool = Field(
        False,
        description=(
            "The host serves many unrelated listed URLs (docs.google.com, URL shorteners): "
            "only an exact listing is evidence there, not the host."
        ),
    )
    stale: bool = Field(False, description="A feed snapshot is older than the allowed age.")
    snapshots: tuple[FeedInfo, ...] = ()


class UrlArguments(Arguments):
    url: str = Field(min_length=1, max_length=2048, description="A URL from the email.")


def lookup_reputation_tool(
    index: ReputationIndex | None, max_age_days: int = 7
) -> Tool[UrlArguments, Reputation]:
    async def lookup_reputation(case: Case, arguments: UrlArguments) -> Reputation:
        url = arguments.url.strip()
        if index is None or not index.feeds:
            return Reputation(url=clip(url, 300) or "", available=False, listed=False)
        key = url_key(url)
        listings = index.by_url.get(key, []) if key else []
        host = _host_of(key) if key else ""
        same_host = index.by_host.get(host, 0)
        return Reputation(
            url=clip(url, 300) or "",
            available=True,
            listed=bool(listings),
            listings=tuple(
                ListingOut(feed=x.feed, target=clip(x.target, 60), added=x.added)
                for x in listings[:5]
            ),
            same_host_listings=same_host,
            shared_platform=same_host >= SHARED_HOST_LISTINGS or host in SHORTENERS,
            stale=any(f.age_days > max_age_days for f in index.feeds),
            snapshots=tuple(index.feeds),
        )

    return Tool(
        name="lookup_reputation",
        description=(
            "Whether a URL from the email is a known phishing URL, in local snapshots of the "
            "PhishTank and OpenPhish feeds. Nothing is sent anywhere. Not being listed proves "
            "little: feeds lag new campaigns."
        ),
        arguments=UrlArguments,
        run=lookup_reputation,
    )


def network_tools(
    rdap: RdapClient, reputation: ReputationIndex | None, max_age_days: int = 7
) -> Sequence[Tool[Any, Any]]:
    return (domain_age_tool(rdap), lookup_reputation_tool(reputation, max_age_days))
