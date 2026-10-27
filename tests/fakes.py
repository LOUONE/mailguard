"""Synthetic corpus files for tests: same formats as the real archives, made-up messages."""

from __future__ import annotations

import io
import mailbox
import tarfile
from pathlib import Path

from mailguard.data.download import target_path
from mailguard.data.sources import Corpus, Label, Source


def source(filename: str, label: Label = "ham") -> Source:
    corpus: Corpus = "spamassassin" if filename.endswith(".tar.bz2") else "nazario"
    return Source(corpus, filename, label, f"https://example.org/{filename}", "0" * 64)


def write_tar(dest: Path, src: Source, members: dict[str, bytes]) -> None:
    path = target_path(src, dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:bz2") as archive:
        directory = tarfile.TarInfo("easy_ham")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, data in members.items():
            info = tarfile.TarInfo(f"easy_ham/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def write_mbox(dest: Path, src: Source, messages: list[bytes]) -> None:
    path = target_path(src, dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    box = mailbox.mbox(path)
    for raw in messages:
        box.add(raw)
    box.close()
