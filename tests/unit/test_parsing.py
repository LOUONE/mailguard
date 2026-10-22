from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.message import EmailMessage

import pytest
from pydantic import ValidationError

from mailguard.parsing import Attachment, Email, parse_email

# All fixtures are synthetic: no message from the training corpora is committed.

TEXT_ONLY = b"""\
Return-Path: <bounce@mailer.example.net>
Received: from mx1.example.net by mx.example.org; Tue, 14 Oct 2025 09:12:03 +0000
Received: from mailer.example.net by mx1.example.net; Tue, 14 Oct 2025 09:12:01 +0000
From: "Example Bank" <Security@Example-Bank.com>
Reply-To: helpdesk@evil.example
To: alice@example.org, Bob <BOB@example.org>
Subject: Verify your account
Date: Tue, 14 Oct 2025 09:12:00 +0000
Message-ID: <abc123@example-bank.com>
Content-Type: text/plain; charset="utf-8"

Your account is locked. Visit https://example-bank.com.verify.example/login.
Or www.example-bank.com/help, today.
"""

MULTIPART_ALTERNATIVE = b"""\
From: news@example.org
To: alice@example.org
Subject: Newsletter
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="BOUNDARY"

--BOUNDARY
Content-Type: text/plain; charset=utf-8

Read online: https://example.org/n/42
--BOUNDARY
Content-Type: text/html; charset=utf-8

<html><body><a href="https://example.org/n/42">Read online</a>
<a href="https://track.example.net/c?id=1&amp;u=2">Unsubscribe</a></body></html>
--BOUNDARY--
"""

HTML_ONLY = b"""\
From: support@example.org
Subject: Action required
Content-Type: text/html; charset=iso-8859-1
Content-Transfer-Encoding: quoted-printable

<p>Ch=E8re cliente, <a href=3D"http://198.51.100.7/paypal/">cliquez ici</a></p>
"""


def build_with_attachment() -> bytes:
    msg = EmailMessage()
    msg["From"] = "hr@example.org"
    msg["Subject"] = "Invoice"
    msg.set_content("See attached.")
    msg.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="inv.pdf")
    return msg.as_bytes()


# --- message shapes -------------------------------------------------------------


def test_text_only_email() -> None:
    email = parse_email(TEXT_ONLY)

    assert email.subject == "Verify your account"
    assert email.from_name == "Example Bank"
    assert email.from_address == "security@example-bank.com"
    assert email.reply_to == "helpdesk@evil.example"
    assert email.return_path == "bounce@mailer.example.net"
    assert email.to == ("alice@example.org", "bob@example.org")
    assert email.message_id == "<abc123@example-bank.com>"
    assert email.date == datetime(2025, 10, 14, 9, 12, tzinfo=UTC)
    assert "account is locked" in email.text_body
    assert email.html_body == ""
    assert email.urls == (
        "https://example-bank.com.verify.example/login",
        "www.example-bank.com/help",
    )
    assert email.attachments == ()
    assert email.defects == ()


def test_headers_keep_order_and_repeats() -> None:
    email = parse_email(TEXT_ONLY)

    received = email.header_values("received")
    assert len(received) == 2
    assert received[0].startswith("from mx1.example.net")
    assert email.headers[0].name == "Return-Path"


def test_multipart_alternative_keeps_both_bodies() -> None:
    email = parse_email(MULTIPART_ALTERNATIVE)

    assert email.text_body.strip() == "Read online: https://example.org/n/42"
    assert '<a href="https://example.org/n/42">' in email.html_body
    # Same URL in text and HTML is reported once; HTML entities are decoded.
    assert email.urls == ("https://example.org/n/42", "https://track.example.net/c?id=1&u=2")


def test_html_only_email() -> None:
    email = parse_email(HTML_ONLY)

    assert email.text_body == ""
    assert "Chère cliente" in email.html_body
    assert email.urls == ("http://198.51.100.7/paypal/",)


def test_attachment_metadata_only() -> None:
    email = parse_email(build_with_attachment())

    assert email.text_body.strip() == "See attached."
    assert email.attachments == (
        Attachment(
            filename="inv.pdf",
            content_type="application/pdf",
            size=13,
            sha256=hashlib.sha256(b"%PDF-1.4 fake").hexdigest(),
        ),
    )


def test_unnamed_binary_part_counts_as_attachment() -> None:
    raw = (
        b"Content-Type: multipart/mixed; boundary=B\n\n"
        b"--B\nContent-Type: text/plain\n\nhi\n"
        b"--B\nContent-Type: application/octet-stream\n\n\x00\x01\n"
        b"--B--\n"
    )

    email = parse_email(raw)

    assert [a.content_type for a in email.attachments] == ["application/octet-stream"]
    assert email.attachments[0].filename is None


# --- decoding -------------------------------------------------------------------


def test_encoded_words_in_headers() -> None:
    raw = (
        b"From: =?iso-8859-1?q?Andr=E9_Dupont?= <andre@example.fr>\n"
        b"Subject: =?utf-8?b?Q29tcHRlIGJsb3F1w6k=?=\n\nx"
    )

    email = parse_email(raw)

    assert email.subject == "Compte bloqué"
    assert email.from_name == "André Dupont"


def test_unknown_charset_falls_back() -> None:
    raw = (
        b"Subject: =?x-unknown?q?caf=C3=A9?=\n"
        b"Content-Type: text/plain; charset=x-unknown\n\ncaf\xc3\xa9"
    )

    email = parse_email(raw)

    assert email.subject == "café"
    assert email.text_body == "café"


def test_raw_8bit_header() -> None:
    email = parse_email(b"Subject: caf\xc3\xa9\n\nbody")

    assert email.subject == "café"


def test_broken_date_is_none() -> None:
    email = parse_email(b"Date: not a date\nSubject: x\n\nbody")

    assert email.date is None
    assert email.subject == "x"


@pytest.mark.parametrize("raw", [b"", b"\n\n", b"just some text, no headers"])
def test_degenerate_input_never_raises(raw: bytes) -> None:
    assert isinstance(parse_email(raw), Email)


# --- model ----------------------------------------------------------------------


def test_email_is_immutable() -> None:
    email = parse_email(TEXT_ONLY)

    with pytest.raises(ValidationError):
        email.subject = "changed"  # type: ignore[misc]


def test_model_validators() -> None:
    email = Email(from_address="  <Alice@Example.ORG> ", to=["A@x.org", "", " "], from_name="  ")

    assert email.from_address == "alice@example.org"
    assert email.to == ("a@x.org",)
    assert email.from_name is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown_field": 1},
        {"attachments": [{"content_type": "x/y", "size": -1, "sha256": "0" * 64}]},
        {"attachments": [{"content_type": "x/y", "size": 1, "sha256": "not-a-hash"}]},
        {"headers": [{"name": "", "value": "v"}]},
    ],
)
def test_model_rejects_invalid_data(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Email(**kwargs)
