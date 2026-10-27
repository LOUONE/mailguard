from __future__ import annotations

import math
import socket

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mailguard.features.psl import PublicSuffixList, public_suffix, registrable_domain
from mailguard.features.urls import (
    URL_FEATURES,
    UrlFeatures,
    anchor_mismatch,
    is_ip_host,
    parse_url,
    shannon_entropy,
    url_features,
)
from mailguard.parsing import parse_email
from mailguard.parsing.html import Anchor

# --- public suffix list ----------------------------------------------------------------------

MINI_LIST = """\
// comment
com
uk
co.uk
*.ck
!www.ck
github.io  trailing text is ignored
"""


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("login.paypal.co.uk", "paypal.co.uk"),
        ("WWW.Example.COM.", "example.com"),
        ("a.b.ck", "a.b.ck"),  # *.ck: b.ck is a public suffix
        ("www.ck", "www.ck"),  # !www.ck: the exception is registrable
        ("user.github.io", "user.github.io"),
        ("x.unknowntld", "x.unknowntld"),  # default rule: unknown TLDs are suffixes
        ("co.uk", None),
        ("com", None),
        ("", None),
        ("a..com", None),
    ],
)
def test_registrable_domain_follows_psl_rules(host: str, expected: str | None) -> None:
    assert PublicSuffixList.parse(MINI_LIST).registrable_domain(host) == expected


def test_bundled_snapshot_knows_real_suffixes() -> None:
    assert registrable_domain("secure.login.bank.co.uk") == "bank.co.uk"
    assert registrable_domain("bücher.de") == "xn--bcher-kva.de"
    assert public_suffix("x.s3.amazonaws.com") is not None
    assert public_suffix("a.example.co.uk") == "co.uk"
    assert public_suffix("a..b") is None


# --- single URLs -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [("198.51.100.7", True), ("2001:db8::1", True), ("3232235777", True), ("0x7f000001", True),
     ("example.org", False), ("", False)],
)  # fmt: skip
def test_ip_hosts_include_integer_and_hex_forms(host: str, expected: bool) -> None:
    assert is_ip_host(host) is expected


def test_parse_url_reads_what_a_browser_would() -> None:
    url = parse_url("http://paypal.com@login.secure.evil.example.co.uk:8080/x")

    assert url is not None
    assert (url.host, url.domain) == ("login.secure.evil.example.co.uk", "example.co.uk")
    assert (url.subdomain_depth, url.has_userinfo, url.port_is_standard) == (3, True, False)


@pytest.mark.parametrize("url", ["http://[::1", "http://", "mailto:a@b.c", "javascript:x()", "  "])
def test_unusable_urls_are_skipped(url: str) -> None:
    assert parse_url(url) is None


def test_bad_port_counts_as_nonstandard() -> None:
    url = parse_url("http://example.org:99999/")

    assert url is not None and not url.port_is_standard


def test_text_urls_without_scheme_are_understood() -> None:
    url = parse_url("www.example.org/login")
    with_port = parse_url("example.org:8080/x")

    assert url is not None and url.domain == "example.org"
    assert with_port is not None and not with_port.port_is_standard


def test_entropy() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


# --- anchors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Anchor("http://198.51.100.7/paypal/", "https://www.paypal.com/signin"), True),
        (Anchor("https://evil.example/", "Bank.com"), True),
        (Anchor("https://www.paypal.com/x", "PAYPAL.COM"), False),
        (Anchor("https://login.bank.co.uk/", "www.bank.co.uk"), False),  # same registrable domain
        (Anchor("https://evil.example/", "Click here"), False),
        (Anchor("http://[::1", "bank.com"), False),
    ],
)
def test_anchor_mismatch(anchor: Anchor, expected: bool) -> None:
    assert anchor_mismatch(anchor) is expected


# --- per-message features --------------------------------------------------------------------


def test_url_features_on_a_hostile_list() -> None:
    features = url_features(
        [
            "http://paypal.com@198.51.100.7:8080/login",
            "www.a.b.c.example.co.uk/x",
            "https://bit.ly/abc",
            "http://xn--pypal-4ve.com",
            "http://x7f.secure-update.top/verify",
            "http://[::1",
        ],
        [Anchor("http://evil.example/x", "www.paypal.com")],
    )

    assert features["url_count"] == 5  # the broken URL is skipped
    assert features["url_ip_hosts"] == 1
    assert features["url_max_subdomain_depth"] == 4  # www.a.b.c
    assert features["url_userinfo"] == features["url_nonstandard_ports"] == 1
    assert features["url_shorteners"] == features["url_punycode_hosts"] == 1
    assert features["url_suspicious_tlds"] == 1
    assert features["url_keyword_hits"] == 2
    assert features["url_anchor_mismatches"] == 1


def test_no_url_gives_zeros() -> None:
    assert set(url_features([]).values()) == {0.0}


@given(st.lists(st.text(max_size=200), max_size=5))
@example(["http://[", "http://a:-1", "http://\x00@￿", "0x", "http://" + "a." * 200 + "com"])
def test_features_are_finite_and_non_negative_for_any_string(urls: list[str]) -> None:
    features = url_features(urls, [Anchor(u, u) for u in urls])

    assert features.keys() == set(URL_FEATURES)
    assert all(math.isfinite(v) and v >= 0 for v in features.values())


def test_transformer_builds_one_row_per_email() -> None:
    emails = [
        parse_email(
            b"Content-Type: text/html\n\n"
            b'<a href="http://198.51.100.7/a">www.bank.com</a> see https://bit.ly/x\n'
        ),
        parse_email(b"Subject: no links\n\nhello\n"),
    ]

    out = UrlFeatures().fit(emails).transform(emails)

    assert out.shape == (2, len(URL_FEATURES))
    assert out.dtype == np.float64
    row = dict(zip(URL_FEATURES, out[0], strict=True))
    # The link text "www.bank.com" is visible text, so the parser lists it as a URL too.
    assert row["url_count"] == 3 and row["url_anchor_mismatches"] == 1
    assert not out[1].any()
    assert list(UrlFeatures().get_feature_names_out()) == list(URL_FEATURES)


def test_extraction_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    registrable_domain.__globals__["default_list"].cache_clear()  # load the snapshot under the ban
    email = parse_email(
        b"Content-Type: text/html\n\n"
        b'<a href="http://login.example.co.uk/a">www.bank.com</a> https://bit.ly/x\n'
    )

    assert UrlFeatures().transform([email]).shape == (1, len(URL_FEATURES))
