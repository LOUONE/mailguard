"""Malformed and unusual messages, each pinned to the behaviour we rely on downstream."""

from __future__ import annotations

import pytest

from mailguard.parsing import parse_email
from mailguard.parsing import parser as parser_module
from mailguard.parsing import urls as urls_module
from mailguard.parsing.urls import urls_in_html, urls_in_text

# --- MIME structure -------------------------------------------------------------


def test_multipart_without_text_plain() -> None:
    raw = (
        b"Content-Type: multipart/mixed; boundary=B\n\n"
        b"--B\nContent-Type: text/html\n\n<p>hi</p>\n"
        b"--B\nContent-Type: application/pdf\nContent-Disposition: attachment; filename=a.pdf\n\n"
        b"PDF\n--B--\n"
    )

    email = parse_email(raw)

    assert email.text_body == ""
    assert email.html_body.strip() == "<p>hi</p>"
    assert [a.filename for a in email.attachments] == ["a.pdf"]


def test_nested_multipart_and_attached_message() -> None:
    raw = (
        b"Content-Type: multipart/mixed; boundary=O\n\n"
        b"--O\nContent-Type: multipart/alternative; boundary=I\n\n"
        b"--I\nContent-Type: text/plain\n\nplain\n"
        b"--I\nContent-Type: text/html\n\n<b>html</b>\n"
        b"--I--\n"
        b"--O\nContent-Type: message/rfc822\n\nSubject: inner\n\ninner body\n"
        b"--O--\n"
    )

    email = parse_email(raw)

    # The text of a forwarded message counts as body text: it is what the reader sees.
    assert email.text_body.split("\n") == ["plain", "inner body"]
    assert email.html_body.strip() == "<b>html</b>"
    assert email.attachments == ()


def test_missing_close_boundary_is_a_defect() -> None:
    raw = b"Content-Type: multipart/mixed; boundary=B\n\n--B\nContent-Type: text/plain\n\nhello\n"

    email = parse_email(raw)

    assert email.text_body.strip() == "hello"
    assert "CloseBoundaryNotFoundDefect" in email.defects


def test_multipart_without_boundary_keeps_its_text() -> None:
    email = parse_email(b"Content-Type: multipart/mixed\n\nVerify your account now\n")

    assert email.text_body.strip() == "Verify your account now"
    assert email.attachments == ()
    assert "NoBoundaryInMultipartDefect" in email.defects


def test_missing_header_body_separator() -> None:
    email = parse_email(b"Subject: hi\nthis is not a header\nmore\n")

    assert email.subject == "hi"
    assert "this is not a header" in email.text_body
    assert "MissingHeaderBodySeparatorDefect" in email.defects


def test_crlf_and_lf_give_the_same_fields() -> None:
    lf = parse_email(b"Subject: hi\nFrom: a@b.org\n\nline1\nline2\n")
    crlf = parse_email(b"Subject: hi\r\nFrom: a@b.org\r\n\r\nline1\r\nline2\r\n")

    assert (crlf.subject, crlf.from_address) == (lf.subject, lf.from_address)
    assert crlf.text_body.replace("\r\n", "\n") == lf.text_body


# --- transfer encodings and charsets ---------------------------------------------


def test_invalid_base64_body_does_not_raise() -> None:
    raw = b"Content-Type: text/plain\nContent-Transfer-Encoding: base64\n\n@@@not base64!!!\n"

    assert "not base64" in parse_email(raw).text_body


def test_quoted_printable_soft_breaks() -> None:
    raw = (
        b"Content-Type: text/plain\nContent-Transfer-Encoding: quoted-printable\n\n"
        b"hello=\n world=3D\n"
    )

    assert parse_email(raw).text_body == "hello world=\n"


@pytest.mark.parametrize(
    ("charset", "body", "expected"),
    [
        (b"utf-7", b"Hi Mom -+Jjo--!", "Hi Mom -☺-!"),
        (b"rot13", b"hello", "hello"),  # a codec, but not a text encoding
        (b"base64", b"hello", "hello"),
        (b"utf-8", b"caf\xe9", "caf\xe9"),  # declared utf-8, actually latin-1
    ],
)
def test_odd_charsets(charset: bytes, body: bytes, expected: str) -> None:
    raw = b"Content-Type: text/plain; charset=" + charset + b"\n\n" + body

    assert parse_email(raw).text_body == expected


def test_invalid_utf8_in_raw_header() -> None:
    assert parse_email(b"Subject: \xff\xfe bad\n\nx").subject == "\xff\xfe bad"


def test_nul_bytes_survive_and_serialise() -> None:
    email = parse_email(b"Subject: a\x00b\n\nx\x00y")

    assert email.text_body == "x\x00y"
    email.model_dump_json()


def test_very_long_header() -> None:
    email = parse_email(b"Subject: " + b"A" * 100_000 + b"\n\nx")

    assert len(email.subject) == 100_000


# --- headers ---------------------------------------------------------------------


def test_missing_headers_give_defaults() -> None:
    email = parse_email(b"\nbody only")

    assert email.headers == ()
    assert (email.subject, email.from_address, email.date, email.to) == ("", None, None, ())


@pytest.mark.parametrize("value", [b"not an address", b"<>", b"@example.org", b"user@"])
def test_from_without_a_domain_is_none(value: bytes) -> None:
    assert parse_email(b"From: " + value + b"\n\nx").from_address is None


def test_uppercase_header_names_and_values() -> None:
    raw = b"CONTENT-TYPE: TEXT/HTML; CHARSET=UTF-8\nfrom: X <A@B.ORG>\n\n<A HREF='HTTPS://EX.ORG'>x</A>"

    email = parse_email(raw)

    assert email.from_address == "a@b.org"
    assert email.urls == ("HTTPS://EX.ORG",)


@pytest.mark.parametrize(
    "value",
    [b"Mon, 1 Jan 99999 00:00:00 +0000", b"Feb 30 2020", b"", b"\xff\xff"],
)
def test_unusable_dates_are_none(value: bytes) -> None:
    assert parse_email(b"Date: " + value + b"\n\nx").date is None


def test_malformed_encoded_word_does_not_raise() -> None:
    assert isinstance(parse_email(b"Subject: =?utf-8?b?###?=\n\nx").subject, str)


# --- safety nets --------------------------------------------------------------------


def test_header_decoder_failure_keeps_raw_value(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(value: object) -> list[tuple[bytes, str | None]]:
        raise ValueError("simulated decode_header failure")

    monkeypatch.setattr(parser_module, "decode_header", explode)

    assert parse_email(b"Subject: =?utf-8?q?hi?=\n\nx").subject == "=?utf-8?q?hi?="


def test_unexpected_crash_returns_partial_email(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(raw: bytes) -> None:
        raise RecursionError("simulated parser crash")

    monkeypatch.setattr(parser_module, "_parse", explode)

    email = parse_email(b"Subject: x\n\nVerify your account")

    assert email.defects == ("ParserFailure:RecursionError",)
    assert "Verify your account" in email.text_body


def test_lone_surrogates_are_replaced() -> None:
    # \udc80-\udcff come from surrogateescape and map back to bytes; \ud800 cannot.
    assert parser_module._scrub("a\udcc3\udca9b") == "aéb"
    assert parser_module._scrub("a\ud800b") == "a?b"


# --- URLs --------------------------------------------------------------------------


def test_trailing_punctuation_is_stripped() -> None:
    text = "See (https://a.example/x), or https://b.example/y. Then https://c.example/z!"

    assert urls_in_text(text) == [
        "https://a.example/x",
        "https://b.example/y",
        "https://c.example/z",
    ]


def test_non_web_hrefs_are_ignored() -> None:
    html = '<a href="mailto:x@y.org">m</a><a href="javascript:alert(1)">j</a><a href="#top">t</a>'

    assert urls_in_html(html) == []


def test_form_action_is_extracted() -> None:
    html = '<form action="https://collect.example/steal"><input name="pwd"></form>'

    assert urls_in_html(html) == ["https://collect.example/steal"]


def test_html_parser_crash_keeps_partial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(self: object, data: str) -> None:
        raise AssertionError("simulated html.parser failure")

    monkeypatch.setattr(urls_module._LinkCollector, "feed", explode)

    assert urls_in_html("<a href='https://x.example'>") == []
