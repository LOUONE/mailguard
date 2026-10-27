"""Header and attachment features that would carry over to a real inbox.

The corpora leave fingerprints in their headers (`spamassassin.taint.org` in 71% of ham,
`monkey.org` and `X-Spam-*` in about half of phishing). Two rules keep them out:

- an allowlist: only `ALLOWED_HEADERS` are read, so the X-Spam-* family, list headers and
  authentication headers cannot reach the model;
- artefact domains substituted by the corpus maintainers count as unknown, never as a domain that
  matches or differs from another.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import ClassVar, Final

from mailguard.features.base import EmailFeatures
from mailguard.features.lists import (
    ARCHIVE_EXTENSIONS,
    ARTEFACT_DOMAINS,
    BRANDS,
    RISKY_EXTENSIONS,
)
from mailguard.features.psl import registrable_domain
from mailguard.parsing import Attachment, Email
from mailguard.parsing.filetypes import EXTENSION_FAMILY, extension_family, extensions

ALLOWED_HEADERS: Final = frozenset({"from", "reply-to", "return-path", "message-id", "date"})

HEADER_FEATURES: Final = (
    "hdr_from_missing",
    "hdr_reply_to_present",
    "hdr_reply_to_mismatch",
    "hdr_return_path_mismatch",
    "hdr_display_name_has_address",
    "hdr_display_name_domain_mismatch",
    "hdr_display_name_brand_mismatch",
    "hdr_message_id_missing",
    "hdr_message_id_domain_mismatch",
    "hdr_date_missing",
    "hdr_defects",
)

# Measured and removed by the leakage audit (docs/features.md): they describe how the corpora
# were collected, not who sent the message.
#  - Received hop count: mostly the recipient's own relays (median 6 hops for ham, 2 for
#    recent phishing).
#  - Recipient count: anonymisation emptied the To addresses of 28% of 2004-2007 phishing.
REMOVED_FEATURES: Final = ("hdr_received_hops", "hdr_recipients")

ATTACHMENT_FEATURES: Final = (
    "att_count",
    "att_risky_extension",
    "att_archive",
    "att_double_extension",
    "att_type_mismatch",
)

_DOMAIN_RE = re.compile(r"(?:[a-z0-9-]+\.)+[a-z]{2,63}", re.IGNORECASE)
_ADDRESS_RE = re.compile(r"[^\s<>@\"']+@[^\s<>@\"']+\.[a-z]{2,63}", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Declared content types whose family can be checked against the sniffed bytes.
_DECLARED_FAMILY: Final[dict[str, str]] = {
    "application/pdf": "pdf",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
    "application/msword": "ole",
    "application/vnd.ms-excel": "ole",
    "application/x-msdownload": "exe",
    "application/x-dosexec": "exe",
    "application/x-rar-compressed": "rar",
    "application/gzip": "gzip",
    "text/html": "html",
}


# --- domains -------------------------------------------------------------------------------


def is_artefact(domain: str) -> bool:
    domain = domain.casefold().rstrip(".")
    return any(domain == a or domain.endswith("." + a) for a in ARTEFACT_DOMAINS)


def domain_of(address_or_host: str | None) -> str | None:
    """Registrable domain of an address (`a@mail.bank.com`) or a host, None if unknown.

    Artefact domains are unknown: they replaced the real ones when the corpus was anonymised.
    """
    if not address_or_host:
        return None
    host = address_or_host.rpartition("@")[2].strip().strip("<>[]").casefold()
    if not host or is_artefact(host):
        return None
    return registrable_domain(host)


def differs(a: str | None, b: str | None) -> bool:
    """Both domains are known and they are not the same. Unknown never counts as a mismatch."""
    return a is not None and b is not None and a != b


# --- display name --------------------------------------------------------------------------


def _without_artefacts(name: str) -> str:
    return " ".join(w for w in name.split() if not any(a in w.casefold() for a in ARTEFACT_DOMAINS))


def display_name_domains(name: str | None) -> set[str]:
    """Registrable domains written inside a display name (`"paypal.com" <x@evil.example>`)."""
    if not name:
        return set()
    found = {domain_of(d) for d in _DOMAIN_RE.findall(_without_artefacts(name))}
    return {d for d in found if d is not None}


def brands_in(name: str | None) -> set[str]:
    """Known brands named in `name`, matched on whole words or runs of up to three words."""
    if not name:
        return set()
    words = _WORD_RE.findall(name.casefold())
    runs = {"".join(words[i : i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    return runs & BRANDS


def brand_mismatch(name: str | None, sender_domain: str | None) -> bool:
    """The display name names a brand that the sender's domain does not contain."""
    brands = brands_in(name)
    if not brands:
        return False
    compact = re.sub(r"[^a-z0-9]", "", sender_domain or "")
    return not any(brand in compact for brand in brands)


# --- features ------------------------------------------------------------------------------


def allowed_view(email: Email) -> Email:
    """The same email with every header outside `ALLOWED_HEADERS` removed."""
    kept = tuple(h for h in email.headers if h.name.casefold() in ALLOWED_HEADERS)
    return email.model_copy(update={"headers": kept})


def header_features(email: Email) -> dict[str, float]:
    email = allowed_view(email)
    from_domain = domain_of(email.from_address)
    message_id = email.message_id or ""
    return {
        "hdr_from_missing": float(email.from_address is None),
        "hdr_reply_to_present": float(email.reply_to is not None),
        "hdr_reply_to_mismatch": float(differs(domain_of(email.reply_to), from_domain)),
        "hdr_return_path_mismatch": float(differs(domain_of(email.return_path), from_domain)),
        "hdr_display_name_has_address": float(
            bool(_ADDRESS_RE.search(_without_artefacts(email.from_name or "")))
        ),
        "hdr_display_name_domain_mismatch": float(
            any(differs(d, from_domain) for d in display_name_domains(email.from_name))
        ),
        "hdr_display_name_brand_mismatch": float(brand_mismatch(email.from_name, from_domain)),
        "hdr_message_id_missing": float(not message_id),
        "hdr_message_id_domain_mismatch": float(
            "@" in message_id and differs(domain_of(message_id.strip("<> ")), from_domain)
        ),
        "hdr_date_missing": float(email.date is None),
        "hdr_defects": float(len(email.defects)),
    }


def type_mismatch(attachment: Attachment) -> bool:
    """The first bytes say one family, the filename or the declared type says another."""
    detected = attachment.detected_type
    if detected is None:
        return False
    claimed = {
        extension_family(attachment.filename),
        _DECLARED_FAMILY.get(attachment.content_type.casefold()),
    } - {None}
    return any(family != detected for family in claimed)


def double_extension(filename: str | None) -> bool:
    """`invoice.pdf.exe`: a known extension hiding in front of a risky or archive one."""
    exts = extensions(filename)
    return (
        len(exts) >= 2
        and exts[-2] in EXTENSION_FAMILY
        and (exts[-1] in RISKY_EXTENSIONS or exts[-1] in ARCHIVE_EXTENSIONS)
    )


def attachment_features(email: Email) -> dict[str, float]:
    attachments = email.attachments
    last_ext = [(extensions(a.filename) or [""])[-1] for a in attachments]
    return {
        "att_count": float(len(attachments)),
        "att_risky_extension": float(sum(e in RISKY_EXTENSIONS for e in last_ext)),
        "att_archive": float(sum(e in ARCHIVE_EXTENSIONS for e in last_ext)),
        "att_double_extension": float(sum(double_extension(a.filename) for a in attachments)),
        "att_type_mismatch": float(sum(map(type_mismatch, attachments))),
    }


class HeaderFeatures(EmailFeatures):
    """Sender consistency and structure, from allowlisted headers only."""

    names: ClassVar[tuple[str, ...]] = HEADER_FEATURES

    def extract(self, email: Email) -> Mapping[str, float]:
        return header_features(email)


class AttachmentFeatures(EmailFeatures):
    """Attachment metadata: counts, risky and double extensions, type mismatches."""

    names: ClassVar[tuple[str, ...]] = ATTACHMENT_FEATURES

    def extract(self, email: Email) -> Mapping[str, float]:
        return attachment_features(email)
