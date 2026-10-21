"""Pinned manifest of the public corpora used for training. See docs/data.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Corpus = Literal["spamassassin", "nazario"]
Label = Literal["ham", "spam", "phishing"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_SPAMASSASSIN_URL = "https://spamassassin.apache.org/old/publiccorpus/"
_NAZARIO_URL = "https://monkey.org/~jose/phishing/"


@dataclass(frozen=True, slots=True)
class Source:
    """One downloadable file, pinned by its SHA-256."""

    corpus: Corpus
    filename: str
    label: Label
    url: str
    sha256: str

    def __post_init__(self) -> None:
        # urllib also opens file:// and ftp:// URLs: only allow HTTPS.
        if not self.url.startswith("https://"):
            raise ValueError(f"{self.filename}: URL must use https, got {self.url!r}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(f"{self.filename}: sha256 must be 64 lowercase hex characters")
        if "/" in self.filename or "\\" in self.filename or self.filename in {"", ".", ".."}:
            raise ValueError(f"invalid filename {self.filename!r}")


def _spamassassin(filename: str, label: Label, sha256: str) -> Source:
    return Source("spamassassin", filename, label, _SPAMASSASSIN_URL + filename, sha256)


def _nazario(filename: str, sha256: str) -> Source:
    return Source("nazario", filename, "phishing", _NAZARIO_URL + filename, sha256)


# Checksums computed on 2026-10-19 from the original hosts.
SOURCES: tuple[Source, ...] = (
    _spamassassin(
        "20030228_easy_ham.tar.bz2",
        "ham",
        "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438",
    ),
    _spamassassin(
        "20030228_easy_ham_2.tar.bz2",
        "ham",
        "b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38",
    ),
    _spamassassin(
        "20030228_hard_ham.tar.bz2",
        "ham",
        "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9",
    ),
    _spamassassin(
        "20030228_spam.tar.bz2",
        "spam",
        "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743",
    ),
    _spamassassin(
        "20050311_spam_2.tar.bz2",
        "spam",
        "44280a0e28bf7645b2279e8e42659271ad82c6f40bf8d364dbda7f1df344a765",
    ),
    _nazario("phishing0.mbox", "6184b0a34ed8cbbb252676e21c1eab9eac5fa4be89c61e30cc9954ebbb25d848"),
    _nazario("phishing1.mbox", "99f02474a11086408a5ce6d11ef38823f003c6611a4312c31fbee96773eeba28"),
    _nazario("phishing2.mbox", "5113277984eae759a5ff958ccf408b542441ce033a49b923ab3b0f324075e6bd"),
    _nazario("phishing3.mbox", "b29336d2e31c2dff19639415e98ed2aced5b4d63675ea5e29bea1a7d4e452841"),
    _nazario("phishing-2015", "7523e62a5e6e4ee8134ab8b1a628e423097a7779e1003b28422864aa34f1c319"),
    _nazario("phishing-2016", "42393e4d686193bfb879d1608c23fd4d3b698ccd4d73f0e1d2276ba709718483"),
    _nazario("phishing-2017", "110fba876b38b44035bf065788952674bc68a7b6879c345a515a9ac844b1d704"),
    _nazario("phishing-2018", "83f49cf2a76c7b29b205d22d8778e46848e36af0f15ee1df126a543696acd48d"),
    _nazario("phishing-2019", "cab39d6fcb38cf80cdaabb3e6ffd1222777e2fcaac7e8480321b7f729c2e3c0e"),
    _nazario("phishing-2020", "99e20bf25f5e8556d37103f2c8140cd75181a62d3df91df8acf9d9aece9f08bb"),
    _nazario("phishing-2021", "636707cff42245d1482e4ae8a2c03369717a5e06bf9d1a0f4ec5dcd846661cee"),
    _nazario("phishing-2022", "a6291e2cb990204766fce425423949b403cce4ae6a660fc8d08812e40a063fa7"),
    _nazario("phishing-2023", "5007f1c1dc91b15e93e988afa3a1cf402e15c88f765d3057be8bfec50dae1fe8"),
    _nazario("phishing-2024", "60afa40a2757170093a2bde31ec9cda4e54d33992203ce04bfbbc86d2e5e5b99"),
    _nazario("phishing-2025", "f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f"),
)
