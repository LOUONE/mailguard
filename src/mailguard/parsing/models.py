"""Validated, immutable representation of a parsed email."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mailguard.parsing.html import read_html


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _normalise_address(value: object) -> object:
    """Lowercase and unwrap an address; anything without a domain becomes None."""
    if isinstance(value, str):
        value = value.strip().strip("<>").strip().lower()
        local, _, domain = value.rpartition("@")
        return value if local and domain else None
    return value


class Header(_Frozen):
    """One header line, decoded to text. Order and repeats are preserved."""

    name: str = Field(min_length=1)
    value: str


class Attachment(_Frozen):
    """Metadata only: the content itself is never kept in memory or logged."""

    filename: str | None = None
    content_type: str
    size: int = Field(ge=0, description="Size in bytes after transfer decoding.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detected_type: str | None = Field(
        default=None, description="Family sniffed from the first bytes (pdf, zip, exe, ...)."
    )


class Email(_Frozen):
    """A raw email reduced to the fields the rest of the pipeline relies on.

    Every field has a safe default, so a message that is empty or badly broken
    still yields a valid `Email`. Problems found while parsing are listed in
    `defects` instead of being raised.
    """

    headers: tuple[Header, ...] = ()
    message_id: str | None = None
    subject: str = ""
    from_name: str | None = None
    from_address: str | None = None
    reply_to: str | None = None
    return_path: str | None = None
    to: tuple[str, ...] = ()
    date: datetime | None = Field(default=None, description="Untrusted: taken from the header.")
    text_body: str = ""
    html_body: str = ""
    urls: tuple[str, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    defects: tuple[str, ...] = ()

    @field_validator("from_address", "reply_to", "return_path", mode="before")
    @classmethod
    def _normalise_single_address(cls, value: object) -> object:
        return _normalise_address(value)

    @field_validator("to", mode="before")
    @classmethod
    def _normalise_address_list(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            cleaned = (_normalise_address(v) for v in value)
            return tuple(v for v in cleaned if v)
        return value

    @field_validator("from_name", "message_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @property
    def body_text(self) -> str:
        """What the reader sees: the plain-text body, or the visible text of the HTML body."""
        if self.text_body.strip():
            return self.text_body
        return read_html(self.html_body).text

    def header_values(self, name: str) -> list[str]:
        """All values of a header, case-insensitively, in message order."""
        wanted = name.lower()
        return [h.value for h in self.headers if h.name.lower() == wanted]
