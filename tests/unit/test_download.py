from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from mailguard.data.download import (
    ChecksumError,
    ensure_source,
    main,
    target_path,
)
from mailguard.data.sources import SOURCES, Source

PAYLOAD = b"From: someone@example.org\n\nhello\n"


def make_source(payload: bytes = PAYLOAD, filename: str = "sample.mbox") -> Source:
    return Source(
        corpus="nazario",
        filename=filename,
        label="phishing",
        url=f"https://example.org/{filename}",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class FakeFetcher:
    """Serves `payload` in small chunks and records which URLs were requested."""

    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url: str) -> Iterator[bytes]:
        self.calls.append(url)
        for i in range(0, len(self.payload), 8):
            yield self.payload[i : i + 8]


def failing_fetcher(url: str) -> Iterator[bytes]:
    yield b"partial"
    raise ConnectionResetError(f"connection lost while fetching {url}")


# --- manifest ---------------------------------------------------------------


def test_manifest_matches_docs() -> None:
    by_label = {label: sum(s.label == label for s in SOURCES) for label in ("ham", "spam")}
    assert len(SOURCES) == 20
    assert by_label == {"ham": 3, "spam": 2}
    assert len({target_path(s, Path()) for s in SOURCES}) == len(SOURCES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "http://example.org/x"),
        ("url", "file:///etc/passwd"),
        ("sha256", "abc"),
        ("sha256", "A" * 64),
        ("filename", "../escape"),
        ("filename", ".."),
    ],
)
def test_source_rejects_invalid_fields(field: str, value: str) -> None:
    fields = {
        "corpus": "nazario",
        "filename": "x.mbox",
        "label": "phishing",
        "url": "https://example.org/x.mbox",
        "sha256": "0" * 64,
        field: value,
    }
    with pytest.raises(ValueError):
        Source(**fields)  # type: ignore[arg-type]


# --- ensure_source ----------------------------------------------------------


def test_first_run_downloads(tmp_path: Path) -> None:
    source, fetch = make_source(), FakeFetcher()

    assert ensure_source(source, tmp_path, fetch=fetch) == "downloaded"

    assert target_path(source, tmp_path).read_bytes() == PAYLOAD
    assert fetch.calls == [source.url]


def test_second_run_is_noop(tmp_path: Path) -> None:
    source, fetch = make_source(), FakeFetcher()
    ensure_source(source, tmp_path, fetch=fetch)

    assert ensure_source(source, tmp_path, fetch=fetch) == "cached"
    assert len(fetch.calls) == 1


def test_wrong_checksum_raises_and_leaves_nothing(tmp_path: Path) -> None:
    source = make_source()

    with pytest.raises(ChecksumError, match=r"checksum mismatch for sample\.mbox"):
        ensure_source(source, tmp_path, fetch=FakeFetcher(b"tampered"))

    assert list(tmp_path.rglob("*.*")) == []


def test_interrupted_download_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(ConnectionResetError):
        ensure_source(make_source(), tmp_path, fetch=failing_fetcher)

    assert list(tmp_path.rglob("*.*")) == []


def test_corrupted_file_on_disk_is_replaced(tmp_path: Path) -> None:
    source, fetch = make_source(), FakeFetcher()
    target = target_path(source, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")

    assert ensure_source(source, tmp_path, fetch=fetch) == "downloaded"
    assert target.read_bytes() == PAYLOAD


def test_force_downloads_again(tmp_path: Path) -> None:
    source, fetch = make_source(), FakeFetcher()
    ensure_source(source, tmp_path, fetch=fetch)

    assert ensure_source(source, tmp_path, fetch=fetch, force=True) == "downloaded"
    assert len(fetch.calls) == 2


# --- CLI ----------------------------------------------------------------------


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "--corpus" in capsys.readouterr().out


def test_cli_downloads_selected_corpus(tmp_path: Path) -> None:
    nazario = make_source(filename="a.mbox")
    spamassassin = Source(
        "spamassassin", "b.tar.bz2", "ham", "https://example.org/b.tar.bz2", nazario.sha256
    )
    fetch = FakeFetcher()

    code = main(
        ["--dest", str(tmp_path), "--corpus", "nazario"],
        sources=[nazario, spamassassin],
        fetch=fetch,
    )

    assert code == 0
    assert fetch.calls == [nazario.url]


def test_cli_reports_failures_and_continues(tmp_path: Path) -> None:
    good = make_source(filename="good.mbox")
    bad = make_source(payload=b"other", filename="bad.mbox")
    fetch = FakeFetcher()

    code = main(["--dest", str(tmp_path)], sources=[bad, good], fetch=fetch)

    assert code == 1
    assert target_path(good, tmp_path).exists()
    assert not target_path(bad, tmp_path).exists()
