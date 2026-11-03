from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from mailguard.agent.llm import ToolCall
from mailguard.agent.tools import Case, Opinion, Toolbox
from mailguard.agent.tools.network import (
    FEEDS,
    MANIFEST,
    AllowlistTransport,
    BlockedHost,
    RdapBootstrap,
    RdapClient,
    ReputationIndex,
    case_domains,
    default_bootstrap,
    network_tools,
    url_key,
)
from mailguard.parsing import parse_email

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 3, tzinfo=UTC)
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
BOOTSTRAP = RdapBootstrap.parse(
    {
        "publication": "2026-10-28T22:00:04Z",
        "services": [
            [["com", "net"], ["http://insecure.example/", "https://rdap.registry.example/com/v1"]],
            [["org"], ["https://rdap.org-registry.example/rdap/"]],
            [["zz"], ["http://only-http.example/"]],
        ],
    }
)
EMAIL = parse_email(
    b"""\
From: "Bank" <alerts@new-bank-login.com>
Reply-To: help@collector.org
Subject: Verify

Log in at https://secure.new-bank-login.com/verify?id=1 or http://198.51.100.7/x
"""
)
CASE = Case(EMAIL, Opinion(score=0.5, probability=0.01, is_phishing=True, threshold=0.3))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here runs with outbound sockets refused: only mocked transports can answer.

    Loopback stays open: the event loop wakes itself through a local socket pair on Windows.
    """
    connect = socket.socket.connect

    def guarded(sock: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            raise OSError("network blocked in tests")
        connect(sock, address)

    def refuse(address: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("network blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "create_connection", refuse)


def rdap_document(registered: str | None = "2026-10-29T08:00:00Z") -> dict[str, Any]:
    events = [{"eventAction": "expiration", "eventDate": "2027-10-29T08:00:00Z"}]
    if registered:
        events.append({"eventAction": "registration", "eventDate": registered})
    return {
        "objectClassName": "domain",
        "status": ["client transfer prohibited"],
        "events": events,
        "entities": [
            {"roles": ["technical"]},
            {
                "roles": ["registrar"],
                "vcardArray": [
                    "vcard",
                    [["version", {}, "text", "4.0"], ["fn", {}, "text", "Cheap Names Ltd"]],
                ],
            },
        ],
    }


def rdap(handler: Callable[[httpx.Request], httpx.Response], timeout_s: float = 5.0) -> RdapClient:
    return RdapClient(
        BOOTSTRAP, timeout_s=timeout_s, transport=httpx.MockTransport(handler), now=lambda: NOW
    )


async def run_tool(
    name: str, client: RdapClient, index: ReputationIndex | None, **args: Any
) -> dict[str, Any]:
    box = Toolbox.of(network_tools(client, index))
    result = await box.call(CASE, ToolCall(id="1", name=name, arguments=args))
    assert result.ok, result.content
    out: dict[str, Any] = json.loads(result.content)
    return out


# --- the fence -------------------------------------------------------------------------------


def test_sockets_really_are_blocked() -> None:
    with pytest.raises(OSError, match="network blocked"):
        socket.create_connection(("example.org", 443))


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/collect",  # not a registry
        "http://rdap.registry.example/com/v1/domain/a.com",  # right host, no TLS
        "https://rdap.registry.example.evil.example/",  # lookalike host
    ],
)
async def test_allowlist_refuses_before_connecting(url: str) -> None:
    reached: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(request)
        return httpx.Response(200)

    transport = AllowlistTransport(BOOTSTRAP.hosts, httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(BlockedHost):
            await http.get(url)
    assert reached == []


async def test_allowlist_lets_registries_through() -> None:
    transport = AllowlistTransport(
        BOOTSTRAP.hosts, httpx.MockTransport(lambda r: httpx.Response(204))
    )
    async with httpx.AsyncClient(transport=transport) as http:
        response = await http.get("https://rdap.registry.example/com/v1/domain/a.com")
    assert response.status_code == 204


def test_bootstrap_keeps_https_servers_only() -> None:
    assert BOOTSTRAP.server_for("A.COM") == "https://rdap.registry.example/com/v1/"
    assert BOOTSTRAP.server_for("x.org") == "https://rdap.org-registry.example/rdap/"
    assert BOOTSTRAP.server_for("x.zz") is None
    assert BOOTSTRAP.server_for("x.de") is None
    assert BOOTSTRAP.hosts == {"rdap.registry.example", "rdap.org-registry.example"}


def test_bundled_bootstrap_covers_common_tlds() -> None:
    bootstrap = default_bootstrap()
    assert bootstrap.publication.startswith("2026-")
    assert all(bootstrap.server_for(f"x.{tld}") for tld in ("com", "org", "net", "uk", "top"))
    assert all(h.count(".") >= 1 for h in bootstrap.hosts)


# --- domain_age ------------------------------------------------------------------------------


def test_case_domains_are_senders_and_links_but_not_ips() -> None:
    assert case_domains(CASE) == {"new-bank-login.com", "collector.org"}


async def test_young_domain_with_its_registrar() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=rdap_document())

    out = await run_tool("domain_age", rdap(handler), None, domain="secure.NEW-BANK-LOGIN.com")
    assert out["status"] == "registered"
    assert out["domain"] == "new-bank-login.com"
    assert out["age_days"] == 4
    assert out["registrar"] == "Cheap Names Ltd"
    assert out["expires_at"] == "2027-10-29T08:00:00Z"
    assert out["statuses"] == ["client transfer prohibited"]
    assert out["source"] == "rdap.registry.example"
    assert str(seen[0].url) == "https://rdap.registry.example/com/v1/domain/new-bank-login.com"
    assert seen[0].headers["accept"] == "application/rdap+json"


async def test_a_domain_not_in_the_email_is_never_looked_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    out = await run_tool("domain_age", rdap(handler), None, domain="attacker-chosen.com")
    assert out["status"] == "not_in_email"


@pytest.mark.parametrize(
    ("response", "status", "detail"),
    [
        (httpx.Response(404), "not_found", None),
        (httpx.Response(503), "unavailable", "HTTP 503"),
        (
            httpx.Response(302, headers={"location": "https://elsewhere.example/"}),
            "unavailable",
            "HTTP 302",
        ),
        (httpx.Response(200, text="<html>"), "unavailable", "not JSON"),
        (httpx.Response(200, json=[1, 2]), "unavailable", "not JSON"),
    ],
)
async def test_registry_answers(response: httpx.Response, status: str, detail: str | None) -> None:
    out = await run_tool("domain_age", rdap(lambda r: response), None, domain="collector.org")
    assert (out["status"], out["detail"]) == (status, detail)


async def test_record_without_dates_or_registrar() -> None:
    document = {
        "events": [{"eventAction": "registration", "eventDate": "garbage"}],
        "entities": [{"roles": ["registrar"]}],
    }
    out = await run_tool(
        "domain_age",
        rdap(lambda r: httpx.Response(200, json=document)),
        None,
        domain="collector.org",
    )
    assert out["status"] == "registered"
    assert (out["registered_at"], out["age_days"], out["registrar"]) == ("garbage", None, None)


async def test_naive_registration_date_is_read_as_utc() -> None:
    client = rdap(lambda r: httpx.Response(200, json=rdap_document("2026-10-01T00:00:00")))
    assert (await client.lookup("collector.org")).age_days == 33


async def test_tld_without_rdap_service() -> None:
    record = await rdap(lambda r: httpx.Response(200)).lookup("bank.de")
    assert (record.status, record.detail) == ("no_rdap_service", ".de")


async def test_transport_failure_and_blocked_redirect_target() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    record = await rdap(refused).lookup("collector.org")
    assert (record.status, record.detail) == ("unavailable", "ConnectError")


async def test_read_timeout() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    record = await rdap(slow).lookup("collector.org")
    assert (record.status, record.detail) == ("unavailable", "ReadTimeout")


async def test_overall_timeout_bounds_a_hanging_registry() -> None:
    class Hanging(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10)
            return httpx.Response(200)

    client = RdapClient(BOOTSTRAP, timeout_s=0.05, transport=Hanging())
    record = await client.lookup("collector.org")
    await client.aclose()
    assert (record.status, record.detail) == ("unavailable", "timeout")


async def test_answers_are_cached_failures_are_not() -> None:
    calls = 0
    answers = [httpx.Response(503), httpx.Response(200, json=rdap_document()), httpx.Response(500)]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return answers[calls - 1]

    clock = [0.0]
    client = RdapClient(
        BOOTSTRAP, transport=httpx.MockTransport(handler), now=lambda: NOW, clock=lambda: clock[0]
    )
    assert (await client.lookup("collector.org")).status == "unavailable"
    assert (await client.lookup("collector.org")).status == "registered"
    assert (await client.lookup("collector.org")).status == "registered"
    assert calls == 2
    clock[0] = 2 * 24 * 3600.0
    assert (await client.lookup("collector.org")).status == "unavailable"
    assert calls == 3


# --- reputation ------------------------------------------------------------------------------

PHISHTANK = """\
phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target
1,https://new-bank-login.com/verify?id=1&amp;s=2,x,2026-11-01T10:00:00+00:00,yes,x,yes,Bank
2,https://secure.new-bank-login.com/verify?id=1,x,2026-11-02T10:00:00+00:00,yes,x,yes,Bank
3,https://docs.google.com/forms/d/abc,x,2026-11-02T11:00:00+00:00,yes,x,yes,Other
4,,x,2026-11-02T11:00:00+00:00,yes,x,yes,Other
"""
OPENPHISH = "https://SECURE.new-bank-login.com/verify?id=1#frag\nhttps://bit.ly/zz\n\n"


def write_feeds(
    directory: Path, downloaded_at: str = "2026-11-02T00:00:00+00:00", **override: bytes
) -> None:
    contents = {"phishtank": PHISHTANK.encode(), "openphish": OPENPHISH.encode()}
    manifest = {}
    for feed in FEEDS:
        data = contents[feed.name]
        (directory / feed.file).write_bytes(override.get(feed.name, data))
        manifest[feed.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "downloaded_at": downloaded_at,
        }
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://Secure.Example.org/a/?q=1&amp;r=2#x", "secure.example.org/a?q=1&r=2"),
        ("http://example.org:80/", "example.org"),
        ("example.org:8443/path/", "example.org:8443/path"),
        ("   ", None),
        ("http://[::1", None),
        ("mailto:", None),
    ],
)
def test_url_key(url: str, key: str | None) -> None:
    assert url_key(url) == key


def test_index_loads_and_counts(tmp_path: Path) -> None:
    write_feeds(tmp_path)
    index = ReputationIndex.load(tmp_path, now=lambda: NOW)
    assert [(f.feed, f.entries, f.age_days) for f in index.feeds] == [
        ("phishtank", 3, 1),
        ("openphish", 2, 1),
    ]
    assert len(index.by_url["secure.new-bank-login.com/verify?id=1"]) == 2
    assert index.by_host["secure.new-bank-login.com"] == 2


def test_index_refuses_a_file_changed_after_download(tmp_path: Path) -> None:
    write_feeds(tmp_path, openphish=b"https://other.example/\n")
    with pytest.raises(ValueError, match=r"openphish.txt does not match"):
        ReputationIndex.load(tmp_path)


def test_index_skips_feeds_missing_from_the_manifest(tmp_path: Path) -> None:
    write_feeds(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    del manifest["phishtank"]
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert [f.feed for f in ReputationIndex.load(tmp_path).feeds] == ["openphish"]


async def test_exact_listing_in_two_feeds(tmp_path: Path) -> None:
    write_feeds(tmp_path)
    index = ReputationIndex.load(tmp_path, now=lambda: NOW)
    out = await run_tool(
        "lookup_reputation",
        rdap(lambda r: httpx.Response(500)),
        index,
        url="http://secure.new-bank-login.com/verify?id=1",
    )
    assert out["available"] and out["listed"]
    assert [(x["feed"], x["target"]) for x in out["listings"]] == [
        ("phishtank", "Bank"),
        ("openphish", None),
    ]
    assert out["same_host_listings"] == 2
    assert out["shared_platform"] is False
    assert out["stale"] is False


async def test_html_escaped_feed_url_still_matches(tmp_path: Path) -> None:
    write_feeds(tmp_path)
    index = ReputationIndex.load(tmp_path, now=lambda: NOW)
    out = await run_tool(
        "lookup_reputation",
        rdap(lambda r: httpx.Response(500)),
        index,
        url="https://new-bank-login.com/verify?id=1&s=2",
    )
    assert out["listed"] is True


async def test_unlisted_url_on_a_shortener_or_a_stale_snapshot(tmp_path: Path) -> None:
    write_feeds(tmp_path, downloaded_at="2026-10-01T00:00:00+00:00")
    index = ReputationIndex.load(tmp_path, now=lambda: NOW)
    out = await run_tool(
        "lookup_reputation", rdap(lambda r: httpx.Response(500)), index, url="https://bit.ly/other"
    )
    assert out["listed"] is False
    assert out["same_host_listings"] == 1
    assert out["shared_platform"] is True
    assert out["stale"] is True


async def test_unusable_url_and_missing_snapshot(tmp_path: Path) -> None:
    write_feeds(tmp_path)
    index = ReputationIndex.load(tmp_path, now=lambda: NOW)
    client = rdap(lambda r: httpx.Response(500))
    weird = await run_tool("lookup_reputation", client, index, url="mailto:")
    assert (weird["listed"], weird["same_host_listings"]) == (False, 0)
    missing = await run_tool("lookup_reputation", client, None, url="https://a.example/")
    assert (missing["available"], missing["listed"]) == (False, False)
    empty = await run_tool("lookup_reputation", client, ReputationIndex(), url="https://a.example/")
    assert empty["available"] is False


def test_outbound_connect_is_blocked_but_loopback_is_not() -> None:
    with socket.socket() as sock, pytest.raises(OSError, match="network blocked"):
        sock.connect(("93.184.215.14", 443))


async def test_record_without_a_registration_event() -> None:
    client = rdap(lambda r: httpx.Response(200, json=rdap_document(registered=None)))
    record = await client.lookup("collector.org")
    assert (record.status, record.registered_at, record.age_days) == ("registered", None, None)
