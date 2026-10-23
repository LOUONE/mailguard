"""Stream raw messages out of the downloaded archives, one at a time."""

from __future__ import annotations

import mailbox
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mailguard.data.download import target_path
from mailguard.data.sources import SOURCES, Source

# SpamAssassin archives ship a `cmds` file next to the messages: it is not mail.
_NOT_MAIL = frozenset({"cmds"})


@dataclass(frozen=True, slots=True)
class RawMessage:
    source: Source
    key: str
    raw: bytes

    @property
    def id(self) -> str:
        """Stable identifier: corpus/file/key."""
        return f"{self.source.corpus}/{self.source.filename}/{self.key}"


def iter_messages(source: Source, dest: Path) -> Iterator[RawMessage]:
    """Yield every message of one downloaded file without extracting anything to disk."""
    path = target_path(source, dest)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `make data` first")

    if source.filename.endswith(".tar.bz2"):
        # Members are read in memory, never extracted, so archive paths cannot
        # write outside data/ (no tar path-traversal risk).
        with tarfile.open(path, mode="r:bz2") as archive:
            for member in archive:
                name = PurePosixPath(member.name).name
                if not member.isfile() or name in _NOT_MAIL:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    yield RawMessage(source, name, handle.read())
    else:
        box = mailbox.mbox(path, create=False)
        try:
            for key in box.iterkeys():
                yield RawMessage(source, str(key), box.get_bytes(key))
        finally:
            box.close()


def iter_corpus(dest: Path, sources: Iterable[Source] = SOURCES) -> Iterator[RawMessage]:
    """Yield every message of every source, in manifest order."""
    for source in sources:
        yield from iter_messages(source, dest)
