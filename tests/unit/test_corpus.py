from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mailguard.data.corpus import iter_corpus, iter_messages
from mailguard.data.dedup import exact_fingerprint, normalise_text, normalised_fingerprint
from mailguard.data.records import iter_records
from tests.fakes import source, write_mbox, write_tar

HAM = b"From: a@example.org\nDate: Tue, 14 Oct 2025 11:00:00 +0200\n\nHello Bob, lunch at 12?\n"
PHISH = (
    b"From: x@example.net\nContent-Type: text/html\n\n"
    b'<p>Your account 4821 is locked. <a href="https://evil.example/a?u=1">Unlock</a></p>\n'
)


# --- corpus reader --------------------------------------------------------------


def test_tar_archive_skips_directories_and_cmds(tmp_path: Path) -> None:
    src = source("ham.tar.bz2")
    write_tar(tmp_path, src, {"00001.abc": HAM, "cmds": b"mv 1 2\n"})

    messages = list(iter_messages(src, tmp_path))

    assert [(m.key, m.raw) for m in messages] == [("00001.abc", HAM)]
    assert messages[0].id == "spamassassin/ham.tar.bz2/00001.abc"


def test_mbox_yields_each_message(tmp_path: Path) -> None:
    src = source("phishing-2025", "phishing")
    write_mbox(tmp_path, src, [HAM, PHISH])

    messages = list(iter_messages(src, tmp_path))

    assert [m.key for m in messages] == ["0", "1"]
    assert messages[1].raw.startswith(b"From: x@example.net")


def test_iter_corpus_chains_sources_lazily(tmp_path: Path) -> None:
    tar, mbox = source("ham.tar.bz2"), source("phishing-2025", "phishing")
    write_tar(tmp_path, tar, {"1": HAM})
    write_mbox(tmp_path, mbox, [PHISH])

    stream = iter_corpus(tmp_path, [tar, mbox])

    assert next(stream).source == tar  # nothing read beyond what was asked for
    assert [m.source for m in stream] == [mbox]


def test_missing_file_explains_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        next(iter_messages(source("absent.mbox"), tmp_path))


# --- records --------------------------------------------------------------------


def test_records_hold_counts_and_hashes_only(tmp_path: Path) -> None:
    tar, mbox = source("ham.tar.bz2"), source("phishing-2025", "phishing")
    write_tar(tmp_path, tar, {"1": HAM})
    write_mbox(tmp_path, mbox, [PHISH])

    ham, phish = iter_records(tmp_path, [tar, mbox])

    assert (ham.label, ham.has_text, ham.has_html, ham.n_urls) == ("ham", True, False, 0)
    assert ham.date == datetime(2025, 10, 14, 9, 0, tzinfo=UTC)
    assert (phish.label, phish.has_html, phish.n_urls, phish.date) == ("phishing", True, 1, None)
    assert phish.body_chars == len("Your account 4821 is locked. Unlock")
    # Only booleans, numbers, short identifiers, hashes and dates: no message content.
    for value in (*vars_of(ham), *vars_of(phish)):
        assert not isinstance(value, str) or ("lunch" not in value and "locked" not in value)


def vars_of(record: object) -> list[object]:
    return [getattr(record, name) for name in record.__slots__]  # type: ignore[attr-defined]


# --- fingerprints ---------------------------------------------------------------


def test_normalisation_ignores_case_numbers_urls_and_spacing() -> None:
    a = "Your account 4821 is LOCKED.\n\nVisit https://evil.example/a?u=1 now!"
    b = "your account 9 is locked visit http://other.example now"

    assert normalise_text(a) == normalise_text(b) == "your account 0 is locked visit url now"
    assert normalised_fingerprint(a) == normalised_fingerprint(b)
    assert exact_fingerprint(a) != exact_fingerprint(b)


@pytest.mark.parametrize("text", ["", "   \n", "!!! ---"])
def test_empty_bodies_have_no_fingerprint(text: str) -> None:
    assert normalised_fingerprint(text) is None


def test_exact_fingerprint_ignores_outer_whitespace_only() -> None:
    assert exact_fingerprint("  hi \n") == exact_fingerprint("hi")
    assert exact_fingerprint("hi") != exact_fingerprint("Hi")
    assert exact_fingerprint(" ") is None


@given(st.text())
def test_normalisation_is_idempotent(text: str) -> None:
    once = normalise_text(text)

    assert normalise_text(once) == once
