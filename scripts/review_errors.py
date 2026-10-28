"""Print the errors of a registered model at its 1% FPR threshold, for manual review.

Local use only: the output contains (scrubbed) message text and must never be committed or pasted
into an issue. What gets committed is the taxonomy built from it: categories and counts, in
docs/error_analysis.md.

    uv run python scripts/review_errors.py --model lightgbm --kind fp
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import numpy as np

from mailguard.config import get_settings
from mailguard.data.dataset import load_dataset, load_manifest
from mailguard.features.text import scrubbed_text
from mailguard.models.evaluate import Rows, load_registered, parse_rows, predict, val_thresholds
from mailguard.models.explain import explain
from mailguard.models.pipelines import MODEL_NAMES

EXCERPT = 500


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review_errors")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lightgbm")
    parser.add_argument("--kind", choices=["fp", "fn"], required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
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
        wrong = flagged & (rows.target == 0) if args.kind == "fp" else ~flagged & (rows.target == 1)
        for i in np.flatnonzero(wrong):
            count += 1
            record = rows.frame.iloc[i]
            label, period = record["label"], record["period"]
            print(f"=== #{count} {split} {label} {period} score={scores[i]:.3f}")
            print("subject:", emails[i].subject[:120])
            print("text:", " ".join(scrubbed_text(emails[i]).split())[:EXCERPT])
            reasons = explain(model.pipeline, emails[i], top=4)
            print("top:", ", ".join(f"{r['feature']} {r['contribution']:+.2f}" for r in reasons))
    print(f"--- {count} {args.kind} at threshold {threshold:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
