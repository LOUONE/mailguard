"""Property-based tests: the parser must hold up on input nobody thought of."""

from __future__ import annotations

from email.message import EmailMessage

from hypothesis import example, given
from hypothesis import strategies as st

from mailguard.parsing import Email, parse_email
from mailguard.parsing.urls import urls_in_html, urls_in_text
from tests.strategies import email_like


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
