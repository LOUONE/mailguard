from __future__ import annotations

from email.message import EmailMessage

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mailguard.features.auth import (
    AuthResults,
    authentication_results,
    parse_authentication_results,
)
from mailguard.features.headers import (
    ALLOWED_HEADERS,
    ATTACHMENT_FEATURES,
    HEADER_FEATURES,
    REMOVED_FEATURES,
    AttachmentFeatures,
    HeaderFeatures,
    attachment_features,
    brand_mismatch,
    brands_in,
    display_name_domains,
    domain_of,
    double_extension,
    header_features,
    type_mismatch,
)
from mailguard.parsing import Attachment, parse_email
from mailguard.parsing.filetypes import extension_family, extensions, sniff

# All addresses and domains below are synthetic (RFC 2606 names or made-up brands).

SPOOF = b"""\
From: "PayPal Security <service@paypal.com>" <alerts@secure-mail.example>
Reply-To: help@other.example
Return-Path: <bounce@bulk.example>
Message-ID: <123@mailer.example>
To: victim@example.org
Subject: Your account
X-Spam-Status: Yes, score=12.0
Authentication-Results: mx.example.org; spf=fail smtp.mailfrom=secure-mail.example

Verify now.
"""

CLEAN = b"""\
From: Alice <alice@mail.example.org>
Reply-To: alice@example.org
Return-Path: <alice@example.org>
Message-ID: <abc@mx.example.org>
Date: Tue, 14 Oct 2025 09:12:00 +0000
To: bob@example.org

Lunch?
"""


def features(raw: bytes) -> dict[str, float]:
    return header_features(parse_email(raw))


# --- domains and display names -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a@mail.bank.co.uk", "bank.co.uk"),
        ("<A@Example.ORG>", "example.org"),
        ("mx.example.org", "example.org"),
        ("jm@spamassassin.taint.org", None),  # corpus artefact: unknown, not a domain
        ("x@lists.taint.org", None),
        ("jose@monkey.org", None),
        ("localhost", None),  # a bare suffix has no registrable domain
        ("", None),
        (None, None),
    ],
)
def test_domain_of(value: str | None, expected: str | None) -> None:
    assert domain_of(value) == expected


def test_display_name_domains_ignore_artefacts() -> None:
    assert display_name_domains('"service@PayPal.com" via www.bank.co.uk') == {
        "paypal.com",
        "bank.co.uk",
    }
    assert display_name_domains("jose@monkey.org") == set()
    assert display_name_domains(None) == set()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PayPal Security", {"paypal"}),
        ("Bank of America Alerts", {"bankofamerica"}),
        ("Wells-Fargo", {"wellsfargo"}),
        ("First Bank", set()),  # "irs" inside "first" must not match
        (None, set()),
    ],
)
def test_brands_match_whole_words(name: str | None, expected: set[str]) -> None:
    assert brands_in(name) == expected


def test_brand_mismatch_needs_a_brand_missing_from_the_domain() -> None:
    assert brand_mismatch("PayPal Service", "secure-mail.example")
    assert brand_mismatch("PayPal Service", None)
    assert not brand_mismatch("PayPal", "paypal.co.uk")
    assert not brand_mismatch("Alice", "example.org")


# --- header features -----------------------------------------------------------------------


def test_spoofed_sender_lights_up() -> None:
    f = features(SPOOF)

    assert f["hdr_reply_to_present"] == f["hdr_reply_to_mismatch"] == 1
    assert f["hdr_return_path_mismatch"] == 1
    assert f["hdr_display_name_has_address"] == 1
    assert f["hdr_display_name_domain_mismatch"] == 1
    assert f["hdr_display_name_brand_mismatch"] == 1
    assert f["hdr_message_id_domain_mismatch"] == 1
    assert f["hdr_date_missing"] == 1
    assert f["hdr_from_missing"] == f["hdr_message_id_missing"] == 0


def test_consistent_sender_stays_quiet() -> None:
    f = features(CLEAN)

    assert f["hdr_reply_to_present"] == 1
    assert sum(f.values()) == 1  # same registrable domain everywhere


def test_artefact_domains_never_count_as_a_mismatch() -> None:
    raw = (
        b"From: Justin <jm@spamassassin.taint.org>\n"
        b"Return-Path: <list-admin@lists.example.net>\n"
        b"Reply-To: jose@monkey.org\n"
        b"Message-ID: <1@spamassassin.taint.org>\n\nhi\n"
    )

    f = features(raw)

    assert f["hdr_return_path_mismatch"] == f["hdr_reply_to_mismatch"] == 0
    assert f["hdr_message_id_domain_mismatch"] == 0


def test_empty_message() -> None:
    f = features(b"")

    assert f["hdr_from_missing"] == f["hdr_message_id_missing"] == f["hdr_date_missing"] == 1
    assert f.keys() == set(HEADER_FEATURES)


_other_names = st.one_of(
    st.sampled_from(
        ["X-Spam-Status", "X-Spam-Flag", "Delivered-To", "List-Id", "Received", "X-Mailer",
         "Authentication-Results", "DKIM-Signature", "Received-SPF", "X-Original-To", "Cc"]
    ),
    st.from_regex(r"X-[A-Za-z]{1,12}", fullmatch=True),
)  # fmt: skip
_values = st.text(st.characters(min_codepoint=32, max_codepoint=126), max_size=60)


@given(st.lists(st.tuples(_other_names, _values), max_size=8))
def test_headers_outside_the_allowlist_never_change_the_features(
    extra: list[tuple[str, str]],
) -> None:
    lines = b"".join(f"{name}: {value}\n".encode() for name, value in extra)
    head, _, body = SPOOF.partition(b"\n\n")

    assert features(head + b"\n" + lines + b"\n" + body) == features(SPOOF)
    assert all(name.casefold() not in ALLOWED_HEADERS for name, _ in extra)


def test_model_features_exclude_authentication_and_removed_artefacts() -> None:
    for name in HEADER_FEATURES:
        assert not any(word in name for word in ("spf", "dkim", "dmarc", "auth"))
    assert not set(REMOVED_FEATURES) & set(HEADER_FEATURES)


# --- authentication results (for the agent, not the model) ---------------------------------


def test_parse_authentication_results() -> None:
    value = (
        "mx.example.org;\n spf=pass smtp.mailfrom=a.example;\n"
        " dkim=fail (dkim=pass in a comment) header.d=a.example; dkim=pass; DMARC=Fail"
    )

    assert parse_authentication_results(value) == AuthResults(
        "mx.example.org", "pass", "fail", "fail"
    )


def test_missing_methods_are_none() -> None:
    results = parse_authentication_results("mx.example.org; none")

    assert [results.get(m) for m in ("spf", "dkim", "dmarc")] == [None, None, None]


def test_only_the_trusted_server_counts() -> None:
    email = parse_email(
        b"Authentication-Results: mx.example.org; spf=fail\n"
        b"Authentication-Results: forged.example; spf=pass; dkim=pass\n\nhi\n"
    )

    assert authentication_results(email).spf == "fail"  # topmost by default
    assert authentication_results(email, "FORGED.example").dkim == "pass"
    assert authentication_results(email, "other.example") == AuthResults()


# --- attachments ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "family"),
    [
        (b"%PDF-1.7", "pdf"),
        (b"PK\x03\x04rest", "zip"),
        (b"MZ\x90\x00", "exe"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole"),
        (b"\x89PNG\r\n\x1a\n", "image"),
        (b"\xef\xbb\xbf  <!DOCTYPE html><html>", "html"),
        (b"hello", None),
        (b"", None),
    ],
)
def test_sniff(payload: bytes, family: str | None) -> None:
    assert sniff(payload) == family


def test_extensions() -> None:
    assert extensions("Invoice.PDF.exe") == ["pdf", "exe"]
    assert extensions("dir\\scan.docx. ") == ["docx"]
    assert extensions("README") == []
    assert extensions(None) == []
    assert extension_family("x.docm") == "zip"
    assert extension_family("x.unknown") is None


def attachment(filename: str | None, content_type: str, detected: str | None) -> Attachment:
    return Attachment(
        filename=filename,
        content_type=content_type,
        size=1,
        sha256="0" * 64,
        detected_type=detected,
    )


@pytest.mark.parametrize(
    ("att", "expected"),
    [
        (attachment("invoice.pdf", "application/pdf", "exe"), True),
        (attachment("invoice.pdf", "application/octet-stream", "pdf"), False),
        (attachment(None, "application/pdf", "zip"), True),  # declared type disagrees
        (attachment("scan.docx", "application/octet-stream", "zip"), False),
        (attachment("notes.txt", "text/plain", None), False),
    ],
)
def test_type_mismatch(att: Attachment, expected: bool) -> None:
    assert type_mismatch(att) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("invoice.pdf.exe", True), ("photos.jpg.zip", True), ("report.v2.pdf", False),
     ("archive.tar.gz", False), ("setup.exe", False), (None, False)],
)  # fmt: skip
def test_double_extension(filename: str | None, expected: bool) -> None:
    assert double_extension(filename) is expected


def build_with_attachments() -> bytes:
    msg = EmailMessage()
    msg["From"] = "a@example.org"
    msg.set_content("See attached.")
    msg.add_attachment(
        b"MZ\x90\x00 fake", maintype="application", subtype="pdf", filename="inv.pdf"
    )
    msg.add_attachment(b"PK\x03\x04", maintype="application", subtype="zip", filename="a.pdf.zip")
    msg.add_attachment(b"<html>x", maintype="text", subtype="html", filename="login.html")
    return msg.as_bytes()


def test_attachment_features_end_to_end() -> None:
    f = attachment_features(parse_email(build_with_attachments()))

    assert f == {
        "att_count": 3,
        "att_risky_extension": 1,  # .html
        "att_archive": 1,
        "att_double_extension": 1,
        "att_type_mismatch": 1,  # an executable named .pdf
    }


def test_transformers_shapes_and_names() -> None:
    emails = [parse_email(SPOOF), parse_email(build_with_attachments())]

    headers = HeaderFeatures().fit_transform(emails)
    attachments = AttachmentFeatures().fit_transform(emails)

    assert headers.shape == (2, len(HEADER_FEATURES))
    assert attachments.shape == (2, len(ATTACHMENT_FEATURES))
    assert np.isfinite(headers).all() and (headers >= 0).all()
    assert list(HeaderFeatures().get_feature_names_out()) == list(HEADER_FEATURES)
