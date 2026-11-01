"""Test doubles: synthetic corpus files in the real archive formats, and a scripted LLM."""

from __future__ import annotations

import io
import mailbox
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mailguard.agent.llm import JsonSchema, LLMError, Message, Reply, ToolSpec
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


@dataclass
class Call:
    """What a `ScriptedLLM` was asked."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    schema: JsonSchema | None


@dataclass
class ScriptedLLM:
    """An `LLMClient` that plays back replies (or raises errors) in order and records each call.

    Running out of script is a test bug, so it fails loudly instead of inventing an answer.
    """

    script: list[Reply | LLMError]
    model: str = "scripted"
    calls: list[Call] = field(default_factory=list)
    closed: bool = False

    @classmethod
    def saying(cls, *contents: str) -> ScriptedLLM:
        return cls([Reply(message=Message(role="assistant", content=c)) for c in contents])

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        schema: JsonSchema | None = None,
    ) -> Reply:
        self.calls.append(Call(tuple(messages), tuple(tools), schema))
        if not self.script:
            raise AssertionError(f"ScriptedLLM called {len(self.calls)} times, script exhausted")
        step = self.script.pop(0)
        if isinstance(step, LLMError):
            raise step
        return step

    async def aclose(self) -> None:
        self.closed = True
