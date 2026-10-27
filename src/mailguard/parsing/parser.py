"""Raw RFC 5322 bytes -> validated `Email`. Never raises on malformed input."""

from __future__ import annotations

import hashlib
from datetime import datetime
from email import policy
from email.header import decode_header
from email.message import Message
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime

from mailguard.parsing.filetypes import sniff
from mailguard.parsing.models import Attachment, Email, Header
from mailguard.parsing.urls import dedupe, urls_in_html, urls_in_text

# compat32 is the most lenient policy: malformed headers come back as raw
# strings and problems are recorded as defects. The modern `policy.default`
# parses headers eagerly and can raise on hostile input.
_PARSER = BytesParser(policy=policy.compat32)
_BODY_TYPES = frozenset({"text/plain", "text/html"})


def parse_email(raw: bytes) -> Email:
    """Parse one raw email. Malformed input yields a partial `Email`, never an exception."""
    try:
        return _parse(raw)
    except Exception as exc:  # last-resort guard: malformed mail must not crash a pipeline
        return Email(
            text_body=_to_text(raw, None),
            defects=(f"ParserFailure:{type(exc).__name__}",),
        )


def _parse(raw: bytes) -> Email:
    msg = _PARSER.parsebytes(raw)
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[Attachment] = []
    defects: list[str] = [type(d).__name__ for d in msg.defects]

    for part in msg.walk():
        defects.extend(type(d).__name__ for d in part.defects if part is not msg)
        if part.is_multipart():
            continue
        content_type = part.get_content_type()
        if content_type.startswith("multipart/"):
            # A multipart without a usable boundary arrives as one opaque part:
            # keep its content as text instead of filing it as an attachment.
            content_type = "text/plain"
        payload = _payload_bytes(part)
        filename = _filename(part)
        is_body = content_type in _BODY_TYPES and filename is None
        if not is_body or part.get_content_disposition() == "attachment":
            attachments.append(
                Attachment(
                    filename=filename,
                    content_type=content_type,
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    detected_type=sniff(payload),
                )
            )
        elif content_type == "text/plain":
            text_parts.append(_to_text(payload, part.get_content_charset()))
        elif content_type == "text/html":
            html_parts.append(_to_text(payload, part.get_content_charset()))

    text_body = "\n".join(text_parts)
    html_body = "\n".join(html_parts)
    from_name, from_address = _first_address(msg, "From")

    return Email(
        headers=tuple(
            Header(name=name.strip(), value=_decode_header(value))
            for name, value in msg.items()
            if name.strip()
        ),
        message_id=_header(msg, "Message-ID"),
        subject=_header(msg, "Subject") or "",
        from_name=from_name,
        from_address=from_address,
        reply_to=_first_address(msg, "Reply-To")[1],
        return_path=_first_address(msg, "Return-Path")[1],
        to=tuple(addr for _, addr in _addresses(msg, "To")),
        date=_date(msg),
        text_body=text_body,
        html_body=html_body,
        urls=dedupe([*urls_in_text(text_body), *urls_in_html(html_body)]),
        attachments=tuple(attachments),
        defects=tuple(defects),
    )


# --- decoding helpers --------------------------------------------------------


def _to_text(data: bytes, charset: str | None) -> str:
    """Decode with the declared charset, then UTF-8, then latin-1 (which cannot fail)."""
    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except LookupError, UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _scrub(text: str) -> str:
    """Turn lone surrogates (raw 8-bit bytes smuggled through `str`) into real text."""
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")


def _decode_header(value: object) -> str:
    """Decode RFC 2047 encoded words, tolerating unknown charsets and raw 8-bit bytes."""
    try:
        chunks = decode_header(value)  # type: ignore[arg-type]  # str or email.header.Header
    except Exception:
        return _scrub(str(value))
    decoded = (
        _to_text(chunk, charset) if isinstance(chunk, bytes) else str(chunk)
        for chunk, charset in chunks
    )
    return _scrub("".join(decoded)).strip()


def _header(msg: Message, name: str) -> str | None:
    value = msg.get(name)
    return None if value is None else _decode_header(value)


def _addresses(msg: Message, name: str) -> list[tuple[str, str]]:
    values = [_decode_header(v) for v in msg.get_all(name, [])]
    return [(n, a) for n, a in getaddresses(values) if a]


def _first_address(msg: Message, name: str) -> tuple[str | None, str | None]:
    found = _addresses(msg, name)
    return found[0] if found else (None, None)


def _date(msg: Message) -> datetime | None:
    value = _header(msg, "Date")
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except TypeError, ValueError, IndexError, OverflowError:
        return None


def _payload_bytes(part: Message) -> bytes:
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else b""


def _filename(part: Message) -> str | None:
    value = part.get_filename()
    return None if value is None else _decode_header(value) or None
