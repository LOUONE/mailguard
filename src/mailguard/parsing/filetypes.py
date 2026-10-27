"""File-type families from magic bytes and from file extensions. Metadata only.

The parser sniffs the first bytes of each attachment while it hashes it, keeps the family name,
and drops the payload (ADR 0002). Comparing that family with the one implied by the filename
catches `invoice.pdf` that is really an executable.
"""

from __future__ import annotations

from typing import Final

# (prefix, family), checked in order. Office Open XML, JAR and APK files are ZIP archives.
_MAGIC: Final[tuple[tuple[bytes, str], ...]] = (
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),  # empty archive
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole"),  # legacy Office, .msg
    (b"MZ", "exe"),
    (b"\x7fELF", "exe"),
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x1f\x8b", "gzip"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"BM", "image"),
)
_HTML_STARTS: Final = (b"<!doctype html", b"<html", b"<head", b"<body", b"<script")

EXTENSION_FAMILY: Final[dict[str, str]] = {
    "pdf": "pdf",
    **dict.fromkeys(("zip", "docx", "xlsx", "pptx", "docm", "xlsm", "pptm", "jar", "apk"), "zip"),
    **dict.fromkeys(("doc", "xls", "ppt", "msg"), "ole"),
    **dict.fromkeys(("exe", "scr", "dll", "com", "pif", "cpl", "msi"), "exe"),
    "rar": "rar",
    "7z": "7z",
    **dict.fromkeys(("gz", "tgz"), "gzip"),
    **dict.fromkeys(("png", "jpg", "jpeg", "gif", "bmp"), "image"),
    **dict.fromkeys(("html", "htm", "shtml", "xhtml"), "html"),
}


def sniff(payload: bytes) -> str | None:
    """Family of a payload from its first bytes, or None when it is not recognised."""
    for prefix, family in _MAGIC:
        if payload.startswith(prefix):
            return family
    head = payload[:512].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return "html" if head.startswith(_HTML_STARTS) else None


def extensions(filename: str | None) -> list[str]:
    """Lowercased extensions, outermost last: `Invoice.PDF.exe` -> ['pdf', 'exe']."""
    if not filename:
        return []
    name = filename.strip().rstrip(". ").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    parts = name.casefold().split(".")
    return [p.strip() for p in parts[1:] if p.strip()]


def extension_family(filename: str | None) -> str | None:
    """Family implied by the last extension, None if it is unknown or absent."""
    exts = extensions(filename)
    return EXTENSION_FAMILY.get(exts[-1]) if exts else None
