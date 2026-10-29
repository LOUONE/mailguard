"""Append the feature vector of every scored email to a daily JSON Lines file.

What is written: a timestamp, the model version, the score, the label and the structured
features (counts and flags about headers, attachments, URLs and wording). What is never written:
text, addresses, URLs, subjects, TF-IDF columns, request ids. A line cannot be traced back to a
message or to a person; it is what the drift report compares with the validation data.

Writes happen in the request path, so they are one buffered `write` under a lock, without
fsync: losing the last lines in a crash is acceptable for a statistical monitor.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TextIO

PREFIX: Final = "features-"
SUMMARY_FILE: Final = "summary.json"  # written by the drift job, read by the API


class FeatureRecorder:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._day: str | None = None
        self._file: TextIO | None = None

    def record(
        self, model_version: str, score: float, label: str, features: Mapping[str, float]
    ) -> None:
        now = datetime.now(UTC)
        line = json.dumps(
            {
                "time": now.isoformat(timespec="seconds"),
                "model_version": model_version,
                "score": score,
                "label": label,
                "features": dict(features),
            }
        )
        with self._lock:
            self._open_for(now.strftime("%Y-%m-%d")).write(line + "\n")

    def _open_for(self, day: str) -> TextIO:
        if self._file is None or day != self._day:
            self.close_unlocked()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{PREFIX}{day}.jsonl"
            self._file = path.open("a", encoding="utf-8", newline="\n", buffering=1)
            self._day = day
        return self._file

    def close(self) -> None:
        with self._lock:
            self.close_unlocked()

    def close_unlocked(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_records(directory: Path, last: int | None = None) -> list[dict[str, Any]]:
    """The recorded lines, oldest first; only the `last` ones when given. Bad lines are skipped."""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{PREFIX}*.jsonl")):
        records.extend(_lines(path))
    return records[-last:] if last else records


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # a line cut by a crash
                continue
