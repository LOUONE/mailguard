"""The retraining flow: rebuild, train a candidate, compare it with production, promote or keep.

    uv run python scripts/retrain.py                  # full: dataset, train, compare
    uv run python scripts/retrain.py --skip-dataset   # reuse data/processed

Plain functions, one per step, so that any scheduler can run them: cron, a Kubernetes CronJob,
a CI schedule, or the drift job when it flags the traffic. Prefect was planned and is not used:
its embedded server does not run with the FastAPI version the API needs (ADR 0008).

The comparison is only fair on messages neither model has seen. Splits move when data is added
or the parser changes (docs/data.md), so a message in today's test split may have trained the
production model. Every training run records the ids it saw; they are removed from the
evaluation set. A production version without that record is compared only if it was trained on
the same split, and otherwise kept with an explanation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from mailguard.config import Settings, get_settings
from mailguard.data import dataset
from mailguard.data.dataset import load_dataset, load_manifest
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, load_config
from mailguard.models.registry import (
    PRODUCTION,
    Version,
    aliased_version,
    latest_version,
    load_version,
    promote,
    training_record,
)
from mailguard.models.train import (
    calibrate,
    operating_thresholds,
    seen_ids,
    split_data,
    track_run,
    train_baseline,
)
from mailguard.parsing import parse_email
from mailguard.pipeline.promotion import Decision, Evidence, decide, gather

logger = logging.getLogger(__name__)

MODEL: Final = "lightgbm"
OPERATING_POINT: Final = "1pct"
DECISION_TAG: Final = "promotion"


class ComparisonError(Exception):
    """The candidate cannot be compared fairly with production."""


@dataclass(frozen=True, slots=True)
class Outcome:
    candidate: Version
    production: Version | None
    decision: Decision
    evidence: Evidence | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "candidate": self.candidate.version,
            "production": self.production.version if self.production else None,
            "promoted": self.decision.promote,
            "reasons": self.decision.reasons,
            "evidence": self.evidence.as_dict() if self.evidence else None,
        }


def train_candidate(frame: pd.DataFrame, config: BaselineConfig, settings: Settings) -> Version:
    """Train, calibrate and register a new LightGBM version, without promoting it."""
    val = split_data(frame, "val")
    model, _ = calibrate(train_baseline(MODEL, frame, config), val)
    processed = settings.data_dir / "processed"
    track_run(
        MODEL, model, config, {}, processed, settings,
        operating_thresholds(model, val), seen_ids(frame),
    )  # fmt: skip
    return latest_version(settings.model_name, settings)


def evaluation_rows(
    frame: pd.DataFrame,
    current_assignment: str,
    production_assignment: str | None,
    production_seen: set[str] | None,
) -> pd.DataFrame:
    """Test rows that the production model has not seen (the candidate has seen none of them)."""
    test = frame[frame["split"] == "test"]
    if production_seen is not None:
        return test[~test["id"].isin(production_seen)].reset_index(drop=True)
    if production_assignment == current_assignment:
        return test.reset_index(drop=True)
    raise ComparisonError(
        "production was trained on another split and did not record which messages it saw"
    )


def compare(
    rows: pd.DataFrame, production: Version, candidate: Version, settings: Settings, n_boot: int
) -> Evidence:
    emails = [parse_email(raw) for raw in rows["raw"]]
    flags = []
    for version in (production, candidate):
        scores = load_version(version, settings).predict_proba(emails)[:, 1]
        flags.append(scores > version.thresholds[OPERATING_POINT])
    return gather(
        rows["target"].to_numpy(dtype=np.int64),
        rows["label"].eq("ham").to_numpy(dtype=bool),
        rows["cluster"].astype(str).tolist(),
        (flags[0], flags[1]),
        n_boot,
    )


def retrain(settings: Settings, config: BaselineConfig, n_boot: int = 1000) -> Outcome:
    """Train a candidate on data/processed and promote it only on a significant gain."""
    processed = settings.data_dir / "processed"
    frame = load_dataset(processed)
    candidate = train_candidate(frame, config, settings)
    try:
        production = aliased_version(settings.model_name, settings, settings.model_alias)
    except LookupError:
        decision = Decision(True, ["pass: no model in production yet"])
        return conclude(Outcome(candidate, None, decision, None), settings)

    assignment, seen = training_record(production, settings)
    try:
        rows = evaluation_rows(
            frame, load_manifest(processed)["assignment_sha256"], assignment, seen
        )
    except ComparisonError as exc:
        decision = Decision(False, [f"fail: {exc}; retrain it on the current split first"])
        return conclude(Outcome(candidate, production, decision, None), settings)
    logger.info(
        "comparing v%s with v%s on %d messages", candidate.version, production.version, len(rows)
    )
    evidence = compare(rows, production, candidate, settings, n_boot)
    return conclude(Outcome(candidate, production, decide(evidence), evidence), settings)


def conclude(outcome: Outcome, settings: Settings) -> Outcome:
    """Promote or keep, tag the candidate version with the decision, and say why."""
    from mailguard.models.registry import client

    verdict = "promoted" if outcome.decision.promote else "kept production"
    if outcome.decision.promote:
        promote(outcome.candidate, settings, settings.model_alias)
    client(settings).set_model_version_tag(
        outcome.candidate.name, outcome.candidate.version, DECISION_TAG, verdict
    )
    level = logging.INFO if outcome.decision.promote else logging.WARNING
    logger.log(level, "v%s: %s", outcome.candidate.version, verdict)
    for reason in outcome.decision.reasons:
        logger.log(level, "  %s", reason)
    return outcome


def write_outcome(outcome: Outcome, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"retrain-v{outcome.candidate.version}.json"
    path.write_text(json.dumps(outcome.as_dict(), indent=2), encoding="utf-8", newline="\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrain", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-dataset", action="store_true", help="reuse data/processed")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")

    if not args.skip_dataset and dataset.main([]) != 0:
        return 1
    try:
        outcome = retrain(settings, load_config(args.config), args.n_boot)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    path = write_outcome(outcome, settings.data_dir / "retraining")
    logger.info("decision written to %s (alias @%s)", path, PRODUCTION)
    return 0
