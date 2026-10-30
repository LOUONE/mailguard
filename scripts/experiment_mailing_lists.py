"""Issue #73: is the model learning "not a mailing list, so suspicious"? Measured over seeds.

    uv run python scripts/experiment_mailing_lists.py --seeds 5     # about an hour

Three LightGBM variants, each trained on every split:

- `served`: the configuration in production;
- `no_return_path`: without `hdr_return_path_mismatch`, which differs in 61% of training ham
  (a list's bounce address) and 19% of phishing;
- `reweighted`: ham that is not list mail weighted up to count as much as list ham, so the
  model learns as much from personal mail as from lists (a covariate-shift correction);
- each one also judged at a "personal threshold": the 1% FPR target set on validation ham that
  is not list mail, instead of all validation ham.

Writes reports/experiments/mailing_lists.{json,md}: metrics only.
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
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, load_config
from mailguard.models.experiments import Variant, render, run_seed, seeds, summarise
from mailguard.models.slices import from_mailing_list
from mailguard.parsing import parse_email

OUT = Path("reports/experiments")
FEATURE = "hdr_return_path_mismatch"


def without(config: BaselineConfig, feature: str) -> BaselineConfig:
    lightgbm = config.lightgbm
    exclude = (*lightgbm.exclude_features, feature)
    return config.model_copy(
        update={"lightgbm": lightgbm.model_copy(update={"exclude_features": exclude})}
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment_mailing_lists")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")

    frame = load_dataset(settings.data_dir / "processed")
    emails = [parse_email(raw) for raw in frame["raw"]]
    is_list = np.array([from_mailing_list(e) for e in emails], dtype=bool)
    probe = [parse_email(raw) for raw in personal_probe()]
    config = load_config(args.config)
    variants = [
        Variant("served", config),
        Variant("no_return_path", without(config, FEATURE)),
        Variant("reweighted", config, reweight_personal_ham=True),
    ]

    results = []
    for seed in seeds(args.seeds):
        results += run_seed(frame, emails, is_list, probe, seed, variants)
        summary = summarise(results, "served")  # rewritten after each seed: partial runs count
        OUT.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "runs": results}
        (OUT / "mailing_lists.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (OUT / "mailing_lists.md").write_text(
            render(summary, "Mailing-list artefact: variants over split seeds").rstrip() + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
