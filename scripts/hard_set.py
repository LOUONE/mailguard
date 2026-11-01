"""Build the hand-labelled hard evaluation set of the agent (#48). Guideline: docs/hard_set.md.

    uv run python scripts/hard_set.py select      # pick the candidates (once)
    uv run python scripts/hard_set.py annotate    # label them, blind; stop and resume any time
    uv run python scripts/hard_set.py status
    uv run python scripts/hard_set.py export      # write evals/hard_set.csv, print agreement

Local use only for `annotate`: it prints message text to the terminal, links defanged. The work
files live in data/hard_set/ (gitignored); only evals/hard_set.csv is committed: ids, clusters,
labels and categories, no content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from mailguard.agent.hard_set import (
    agreement,
    annotate,
    read_annotations,
    select_candidates,
    write_annotations,
)
from mailguard.config import get_settings
from mailguard.data.dataset import load_dataset, load_manifest
from mailguard.models.evaluate import Rows, predict
from mailguard.models.registry import aliased_version, load_version
from mailguard.parsing import Email, parse_email

OUT: Final = Path("evals/hard_set.csv")
# Wider than any grey zone the agent will get, so each one tried in #49 is a subset (143 test
# clusters with v0.2.0's model, inside the 100-200 the issue asks for).
BAND: Final = (0.03, 0.97)


def work_dir() -> Path:
    return get_settings().data_dir / "hard_set"


def select(force: bool) -> int:
    work = work_dir()
    if read_annotations(work / "annotations.csv") and not force:
        print("error: annotations exist; --force would reorder what is left", file=sys.stderr)
        return 1
    settings = get_settings()
    processed = settings.data_dir / "processed"
    frame = load_dataset(processed)
    assignment = load_manifest(processed)["assignment_sha256"]
    # The served model: it is the one that will decide which emails go to the agent.
    version = aliased_version(settings.model_name, settings, settings.model_alias)
    model = load_version(version, settings)
    threshold = version.thresholds["1pct"]
    test = Rows.of(frame, "split", "test")
    candidates = select_candidates(test.frame, predict(model, test), threshold, BAND)

    work.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(work / "candidates.csv", index=False, lineterminator="\n")
    selection = {
        "date": datetime.now(UTC).date().isoformat(),
        "model": f"{version.name} v{version.version} @{settings.model_alias}",
        "dataset_assignment_sha256": assignment,
        "split": "test",
        "band": list(BAND),
        "threshold_1pct": threshold,
        "clusters": len(candidates),
        "selected_as": candidates["selected_as"].value_counts().to_dict(),
    }
    (work / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return 0


def load_work() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(work_dir() / "candidates.csv", dtype=str)
    frame = load_dataset(get_settings().data_dir / "processed").set_index("id")
    return candidates, frame


def run_annotate() -> int:
    candidates, frame = load_work()
    path = work_dir() / "annotations.csv"

    def email_of(message_id: str) -> Email:
        raw: bytes = frame.at[message_id, "raw"]  # type: ignore[assignment]  # one cell
        return parse_email(raw)

    done = annotate(
        candidates,
        read_annotations(path),
        email_of=email_of,
        ask=input,
        show=print,
        save=lambda annotations: write_annotations(path, annotations),
    )
    print(f"\n{len(done)} / {len(candidates)} annotated, saved in {path}")
    return 0


def status() -> int:
    candidates, _ = load_work()
    done = read_annotations(work_dir() / "annotations.csv")
    print(f"{len(done)} / {len(candidates)} annotated")
    if done:
        counts = pd.Series([f"{a.label}/{a.category}" for a in done]).value_counts()
        print(counts.to_string())
    return 0


def export(partial: bool) -> int:
    candidates, frame = load_work()
    done = read_annotations(work_dir() / "annotations.csv")
    if len(done) < len(candidates) and not partial:
        print(f"error: {len(candidates) - len(done)} not annotated (--partial)", file=sys.stderr)
        return 1
    write_annotations(OUT, sorted(done, key=lambda a: a.id))
    print(f"{OUT}: {len(done)} messages\n")
    print("corpus label (rows) against hand label (columns):")
    corpus_labels = {str(i): str(label) for i, label in frame["label"].items()}
    print(agreement(done, corpus_labels).to_string())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hard_set")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select").add_argument("--force", action="store_true")
    commands.add_parser("annotate")
    commands.add_parser("status")
    commands.add_parser("export").add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # Message text is Unicode; the Windows console is not, by default.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        match args.command:
            case "select":
                return select(args.force)
            case "annotate":
                return run_annotate()
            case "status":
                return status()
            case _:
                return export(args.partial)
    except (FileNotFoundError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
