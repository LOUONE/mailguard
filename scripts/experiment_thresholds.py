"""Issue #85: should duplicate ham clusters count once when the operating threshold is chosen?

    uv run python scripts/experiment_thresholds.py --seeds 8      # about 25 minutes

The served LightGBM, trained on every split seed and judged at three 1% FPR rules: per
validation ham message (today), per duplicate cluster, and on non-list ham only. What matters
here is recall, the per-message ham FPR actually delivered, and how far the threshold itself
moves from one split to the next.

Writes reports/experiments/thresholds.{json,md}: metrics only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mailguard.config import get_settings
from mailguard.data.dataset import load_dataset
from mailguard.data.synthetic import personal_probe
from mailguard.models.config import DEFAULT_CONFIG, load_config
from mailguard.models.experiments import Variant, render, run_seed, seeds, summarise
from mailguard.models.slices import from_mailing_list
from mailguard.parsing import parse_email

OUT = Path("reports/experiments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment_thresholds")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")

    frame = load_dataset(settings.data_dir / "processed")
    emails = [parse_email(raw) for raw in frame["raw"]]
    is_list = np.array([from_mailing_list(e) for e in emails], dtype=bool)
    probe = [parse_email(raw) for raw in personal_probe()]
    variants = [Variant("served", load_config(args.config))]

    results = []
    for seed in seeds(args.seeds):
        results += run_seed(frame, emails, is_list, probe, seed, variants)
        summary = summarise(results, "served")
        OUT.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "runs": results}
        (OUT / "thresholds.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (OUT / "thresholds.md").write_text(
            render(summary, "Operating thresholds over split seeds").rstrip() + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
