"""Property-based tests: the parser must hold up on input nobody thought of."""

from __future__ import annotations

from email.message import EmailMessage

from hypothesis import example, given
from hypothesis import strategies as st

from mailguard.parsing import Email, parse_email
from mailguard.parsing.urls import urls_in_html, urls_in_text

# Building blocks that look like email, so the search reaches deep code paths
# (MIME, encoded words, charsets) instead of stopping at "no headers found".
_header_names = st.sampled_from(
    ["From", "To", "Reply-To", "Return-Path", "Subject", "Date", "Message-ID", "Received",
     "Content-Type", "Content-Transfer-Encoding", "Content-Disposition", "MIME-Version"]
)  # fmt: skip
_header_values = st.one_of(
    st.binary(max_size=80),
    st.sampled_from(
        [
            b"=?utf-8?b?w6k=?=",
            b"=?bogus?q?abc?=",
            b"=?utf-8?b?###?=",
            b"multipart/mixed; boundary=B",
            b"multipart/alternative",
            b"text/html; charset=x-unknown",
            b"text/plain; charset=base64",
            b"base64",
            b"quoted-printable",
            b'attachment; filename="=?utf-8?q?f=C3=A9.pdf?="',
            b"<a@b.c>, , ,<",
            b"Mon, 1 Jan 99999 00:00:00 +0000",
        ]
    ),
)
_header_lines = st.lists(
    st.tuples(_header_names, _header_values).map(lambda kv: kv[0].encode() + b": " + kv[1]),
    max_size=12,
)
_body = st.one_of(
    st.binary(max_size=400),
    st.sampled_from([b"--B\n\nx\n--B--\n", b"--B\nContent-Type: text/html\n\n<a href=x>\n"]),
)
email_like = st.builds(
    lambda headers, sep, body: b"\n".join(headers) + sep + body,
    _header_lines,
    st.sampled_from([b"\n\n", b"\r\n\r\n", b"\n", b""]),
    _body,
)


def assert_clean_parse(raw: bytes) -> Email:
    email = parse_email(raw)
    # The last-resort guard must never be what saved us.
    assert not any(d.startswith("ParserFailure") for d in email.defects), email.defects
    # Every parsed email must be serialisable (the API returns it as JSON).
    email.model_dump_json()
    return email


@given(st.binary(max_size=2000))
@example(b"Subject: caf\xff\xfe\n\nbody")
@example(b"\x00" * 64)
def test_never_fails_on_arbitrary_bytes(raw: bytes) -> None:
    assert_clean_parse(raw)


@given(email_like)
def test_never_fails_on_email_like_bytes(raw: bytes) -> None:
    assert_clean_parse(raw)


_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp")),
    max_size=200,
)


@given(subject=_safe_text, body=_safe_text)
def test_round_trip_well_formed_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "sender@example.org"
    msg.set_content(body)

    email = assert_clean_parse(msg.as_bytes())

    assert " ".join(email.subject.split()) == " ".join(subject.split())
    assert email.text_body.replace("\r\n", "\n").rstrip("\n") == body.rstrip("\n")
    assert email.from_address == "sender@example.org"


@given(st.text(max_size=500))
def test_text_urls_are_well_formed(text: str) -> None:
    for url in urls_in_text(text):
        assert url.lower().startswith(("http://", "https://", "www."))
        assert not any(c.isspace() for c in url)
        assert url in text


@given(st.text(max_size=500))
def test_html_extraction_never_raises(html: str) -> None:
    urls_in_html(html)
