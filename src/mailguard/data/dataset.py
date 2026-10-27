"""The modelling dataset: one row per message, with its target, duplicate cluster and splits.

Splits are decided here, once, before any feature is fitted, so nothing downstream can learn from
validation or test data. Two assignments are stored side by side:

- `split`: the main train / validation / test split, stratified by label (and phishing period)
  and grouped by duplicate cluster, so a campaign never sits on both sides.
- `temporal_split`: a check on phishing only. Phishing up to `TEMPORAL_CUTOFF` trains, later
  phishing is the test set. Ham and spam keep their main split (there is no ham after 2002).

The table holds raw message bytes: it lives in `data/processed/` (gitignored) and is never
committed. Its manifest holds counts and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Literal, get_args

import pandas as pd

from mailguard.config import get_settings
from mailguard.data.corpus import RawMessage, iter_corpus
from mailguard.data.records import to_record
from mailguard.data.sources import SOURCES, Label, Source

Split = Literal["train", "val", "test"]
SPLITS: Final[tuple[Split, ...]] = get_args(Split)
FRACTIONS: Final[Mapping[Split, float]] = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED: Final = 20261027
TEMPORAL_CUTOFF: Final = 2019  # last year on the training side of the temporal check

POSITIVE_LABEL: Final[Label] = "phishing"  # ADR 0001: phishing against ham and spam

DATASET_FILE: Final = "dataset.parquet"
MANIFEST_FILE: Final = "manifest.json"

_YEARLY_FILE_RE = re.compile(r"phishing-(\d{4})")

logger = logging.getLogger(__name__)


# --- per-message metadata ------------------------------------------------------------------


def target(label: Label) -> int:
    """Binary target from ADR 0001: 1 for phishing, 0 for ham and spam."""
    return int(label == POSITIVE_LABEL)


def file_year(source: Source) -> int | None:
    """Year of a yearly Nazario file (`phishing-2021`), None for every other file."""
    match = _YEARLY_FILE_RE.fullmatch(source.filename)
    return int(match.group(1)) if match else None


def period(source: Source) -> str:
    """Coarse period of a file, from the corpus documentation rather than the `Date` header."""
    if source.corpus == "spamassassin":
        return "2002-2003"
    return "2015-2025" if file_year(source) is not None else "2004-2007"


def stratum(label: Label, period_: str) -> str:
    """Stratification key: the label, with phishing split by period so both eras are balanced."""
    return f"{label}:{period_}" if label == POSITIVE_LABEL else label


def cluster_id(message_id: str, normalised_fp: str | None) -> str:
    """Duplicate cluster: the normalised body fingerprint, or a singleton for empty bodies."""
    return normalised_fp if normalised_fp is not None else f"singleton:{message_id}"


def temporal_sides(frame: pd.DataFrame) -> tuple[pd.Series[bool], pd.Series[bool]]:
    """Masks of past and future phishing for the temporal check.

    Past is the 2004-2007 files (no year) and yearly files up to `TEMPORAL_CUTOFF`.
    """
    phishing = frame["label"].eq(POSITIVE_LABEL)
    late = frame["year"].fillna(0).astype(int).gt(TEMPORAL_CUTOFF)
    return phishing & ~late, phishing & late


# --- split assignment --------------------------------------------------------------------


def assign_groups(
    groups: Mapping[str, tuple[str, int]],
    fractions: Mapping[Split, float] = FRACTIONS,
    seed: int = SEED,
) -> dict[str, Split]:
    """Assign each group (cluster) to a split, stratified, keeping every group whole.

    `groups` maps a group id to `(stratum, size)`. Within each stratum, groups are shuffled with
    `seed` and then placed largest first, each into the split furthest below its target size.
    Placing large groups first is what keeps the proportions tight: a 71-message cluster placed
    last could not be balanced any more.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {dict(fractions)}")

    rng = random.Random(seed)
    by_stratum: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for group, (key, size) in sorted(groups.items()):
        by_stratum[key].append((group, size))

    assignment: dict[str, Split] = {}
    for key in sorted(by_stratum):
        members = by_stratum[key]
        rng.shuffle(members)
        members.sort(key=lambda m: m[1], reverse=True)  # stable: equal sizes keep shuffled order
        total = sum(size for _, size in members)
        filled = dict.fromkeys(SPLITS, 0)
        for group, size in members:
            chosen = max(SPLITS, key=lambda s: fractions[s] * total - filled[s])
            assignment[group] = chosen
            filled[chosen] += size
    return assignment


def _majority(values: Iterable[str]) -> str:
    """Most frequent value, ties broken alphabetically so the result never depends on order."""
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def add_splits(
    frame: pd.DataFrame, fractions: Mapping[Split, float] = FRACTIONS, seed: int = SEED
) -> pd.DataFrame:
    """Return a copy of `frame` with `split` and `temporal_split` columns.

    Needs the columns `cluster`, `stratum`, `label` and `year`.
    """
    clusters = frame.groupby("cluster", sort=True)["stratum"]
    groups = {str(c): (_majority(s), len(s)) for c, s in clusters}
    by_cluster = assign_groups(groups, fractions, seed)

    out = frame.copy()
    out["split"] = out["cluster"].map(by_cluster)

    # A cluster holding any past phishing goes to the training side of the temporal check, so a
    # campaign that started before the cutoff never counts as "new" phishing.
    past_rows, future_rows = temporal_sides(out)
    past = set(out.loc[past_rows, "cluster"])
    future = set(out.loc[future_rows, "cluster"])

    def temporal(cluster: str, split: Split) -> Split:
        if cluster in past:
            return "val" if split == "val" else "train"
        if cluster in future:
            return "test"
        return split

    out["temporal_split"] = [
        temporal(c, s) for c, s in zip(out["cluster"], out["split"], strict=True)
    ]
    return out


# --- building ------------------------------------------------------------------------------


def to_row(message: RawMessage) -> dict[str, Any]:
    """One dataset row: the privacy-safe record fields plus target, cluster and raw bytes."""
    record = to_record(message)
    year = file_year(message.source)
    period_ = period(message.source)
    row = asdict(record)
    row.pop("exact_fp")
    normalised_fp = row.pop("normalised_fp")
    row.update(
        target=target(record.label),
        period=period_,
        year=year,
        stratum=stratum(record.label, period_),
        cluster=cluster_id(record.id, normalised_fp),
        raw=message.raw,
    )
    return row


def build_frame(
    dest: Path,
    sources: Iterable[Source] = SOURCES,
    fractions: Mapping[Split, float] = FRACTIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Parse every message under `dest` (the raw data folder) and assign the splits."""
    frame = pd.DataFrame([to_row(m) for m in iter_corpus(dest, sources)])
    if frame.empty:
        raise ValueError(f"no message found under {dest}")
    frame["year"] = frame["year"].astype("Int64")
    return add_splits(frame, fractions, seed).sort_values("id", ignore_index=True)


def assignment_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of every (id, split, temporal split) triple: equal hashes mean equal splits."""
    lines = sorted(
        f"{i}\t{s}\t{t}"
        for i, s, t in zip(frame["id"], frame["split"], frame["temporal_split"], strict=True)
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _counts(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for split, label in zip(frame[column], frame["label"], strict=True):
        out[split][label] = out[split].get(label, 0) + 1
    return out


def build_manifest(
    frame: pd.DataFrame,
    sources: Sequence[Source] = SOURCES,
    fractions: Mapping[Split, float] = FRACTIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Counts and hashes only: safe to print, log or attach to an experiment run."""
    sizes = frame.groupby("cluster").size()
    past, future = temporal_sides(frame)
    crossing = set(frame.loc[past, "cluster"]) & set(frame.loc[future, "cluster"])
    return {
        "rows": len(frame),
        "seed": seed,
        "fractions": dict(fractions),
        "temporal_cutoff": TEMPORAL_CUTOFF,
        "assignment_sha256": assignment_hash(frame),
        "split": _counts(frame, "split"),
        "temporal_split": _counts(frame, "temporal_split"),
        "clusters": {
            "total": len(sizes),
            "with_duplicates": int((sizes > 1).sum()),
            "largest": int(sizes.max()),
            "crossing_temporal_cut": len(crossing),
            "messages_crossing_temporal_cut": int(frame["cluster"].isin(crossing).sum()),
        },
        "sources": {s.filename: s.sha256 for s in sources},
    }


def write_dataset(frame: pd.DataFrame, manifest: Mapping[str, Any], out_dir: Path) -> Path:
    """Write the table and its manifest, each through a temporary file renamed at the end."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / DATASET_FILE
    partial = table.with_name(table.name + ".part")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(table)

    manifest_path = out_dir / MANIFEST_FILE
    partial = manifest_path.with_name(manifest_path.name + ".part")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(manifest_path)
    return table


def load_dataset(processed_dir: Path) -> pd.DataFrame:
    """Read the table written by `write_dataset`."""
    path = processed_dir / DATASET_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `make dataset` first")
    return pd.read_parquet(path)


def load_manifest(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / MANIFEST_FILE
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def manifest_hash(processed_dir: Path) -> str:
    """SHA-256 of the manifest file, recorded with every experiment run."""
    return hashlib.sha256((processed_dir / MANIFEST_FILE).read_bytes()).hexdigest()


# --- command line --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, sources: Sequence[Source] = SOURCES) -> int:
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Parse the downloaded corpora into data/processed/ with grouped splits.",
    )
    parser.add_argument("--raw-dir", type=Path, help="default: $MAILGUARD_DATA_DIR/raw")
    parser.add_argument("--out-dir", type=Path, help="default: $MAILGUARD_DATA_DIR/processed")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(message)s")
    raw_dir: Path = args.raw_dir or settings.data_dir / "raw"
    out_dir: Path = args.out_dir or settings.data_dir / "processed"

    try:
        frame = build_frame(raw_dir, sources, seed=args.seed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    manifest = build_manifest(frame, sources, seed=args.seed)
    path = write_dataset(frame, manifest, out_dir)

    for split, counts in manifest["split"].items():
        logger.info("%-5s %s", split, ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    logger.info("assignment %s", manifest["assignment_sha256"][:12])
    logger.info("%d rows written to %s", manifest["rows"], path)
    return 0
