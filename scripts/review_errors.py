"""Print messages of a registered model at its 1% FPR threshold, for manual review.

Local use only: the output contains (scrubbed) message text and must never be committed or pasted
into an issue. What gets committed is the taxonomy built from it: categories and counts, in
docs/error_analysis.md.

    uv run python scripts/review_errors.py --model lightgbm --kind fp
    uv run python scripts/review_errors.py --kind sample --splits test --n 100

`fp` and `fn` print every error. `sample` prints a seeded random sample of the messages labelled
phishing, flagged or not: reading errors alone finds label noise among misses only, while a
random sample measures how much of the whole phishing label is wrong.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from mailguard.config import get_settings
from mailguard.data.dataset import load_dataset, load_manifest
from mailguard.features.text import scrubbed_text
from mailguard.models.evaluate import Rows, load_registered, parse_rows, predict, val_thresholds
from mailguard.models.explain import explain
from mailguard.models.pipelines import MODEL_NAMES

EXCERPT = 500
SAMPLE_SEED = 20261028


def selected(rows: Rows, flagged: NDArray[np.bool_], kind: str, n: int) -> NDArray[np.intp]:
    """Row indices to print, in row order for errors and in draw order for a sample."""
    if kind == "fp":
        return np.flatnonzero(flagged & (rows.target == 0))
    if kind == "fn":
        return np.flatnonzero(~flagged & (rows.target == 1))
    phishing = np.flatnonzero(rows.is_label("phishing"))
    rng = np.random.default_rng(SAMPLE_SEED)
    drawn: NDArray[np.intp] = rng.choice(phishing, size=min(n, phishing.size), replace=False)
    return drawn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review_errors")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lightgbm")
    parser.add_argument("--kind", choices=["fp", "fn", "sample"], required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--n", type=int, default=100, help="sample size per split (sample only)")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    settings = get_settings()
    processed = settings.data_dir / "processed"
    frame = load_dataset(processed)
    model = load_registered(args.model, settings, load_manifest(processed)["assignment_sha256"])
    val = Rows.of(frame, "split", "val")
    threshold = val_thresholds(val, predict(model, val))["1pct"]

    count = 0
    for split in args.splits:
        rows = Rows.of(frame, "split", split)
        emails = parse_rows(rows)
        scores = predict(model, emails)
        flagged = scores > threshold
        for i in selected(rows, flagged, args.kind, args.n):
            count += 1
            record = rows.frame.iloc[i]
            label, period = record["label"], record["period"]
            state = "flagged" if flagged[i] else "missed"
            print(f"=== #{count} {split} {label} {period} score={scores[i]:.3f} {state}")
            print("subject:", emails[i].subject[:120])
            print("text:", " ".join(scrubbed_text(emails[i]).split())[:EXCERPT])
            reasons = explain(model.pipeline, emails[i], top=4)
            print("top:", ", ".join(f"{r['feature']} {r['contribution']:+.2f}" for r in reasons))
    print(f"--- {count} {args.kind} at threshold {threshold:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
