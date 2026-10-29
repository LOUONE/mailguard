"""First start of the compose stack: download, build, train and promote, unless already done.

    python scripts/bootstrap.py            # what the `bootstrap` service runs
    python scripts/bootstrap.py --smoke    # a tiny synthetic model instead, for CI (seconds)

Idempotent: when a version already holds the production alias it exits at once, so the stack
restarts in seconds after the first run (about 5 minutes: 147 MB of corpora and two models).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from mailguard.config import Settings, get_settings
from mailguard.data import dataset, download
from mailguard.data.dataset import write_dataset
from mailguard.models import train
from mailguard.models.config import BaselineConfig, LightGBMConfig, TextConfig
from mailguard.models.pipelines import lightgbm_pipeline
from mailguard.models.registry import aliased_version, latest_version, promote
from mailguard.parsing import parse_email

logger = logging.getLogger("bootstrap")

# The smoke model: a few made-up messages (RFC 2606 domains), enough to exercise the image.
_HAM = "From: a{i}@example.org\nSubject: lunch\n\nLunch at noon {i}? Bring the slides.\n"
_PHISH = (
    "From: PayPal <x{i}@evil.example>\nReply-To: r@other.example\nSubject: verify your account\n"
    "Content-Type: text/html\n\n"
    '<a href="http://198.51.100.{i}/login">www.paypal.com</a> Verify your account password.\n'
)
_SMOKE = BaselineConfig(
    lightgbm=LightGBMConfig(
        n_estimators=10,
        min_child_samples=1,
        n_jobs=1,
        text=TextConfig(word_max_features=60, char_max_features=60, min_df=1),
    )
)


def has_production(settings: Settings) -> bool:
    try:
        version = aliased_version(settings.model_name, settings, settings.model_alias)
    except LookupError:
        return False
    logger.info("%s v%s is already in production", version.name, version.version)
    return True


def smoke(settings: Settings) -> int:
    """Train and register a tiny model on synthetic messages, as `make train` would."""
    n = 60
    raws = [(_PHISH if i % 2 else _HAM).format(i=i) for i in range(n)]
    target = np.array([i % 2 for i in range(n)], dtype=np.int64)
    split = ["train" if i < 40 else "val" for i in range(n)]
    frame = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "split": split, "raw": raws})
    processed = settings.data_dir / "processed"
    manifest = {"assignment_sha256": "smoke", "rows": n}
    write_dataset(frame.assign(raw=frame["raw"].str.encode("utf-8")), manifest, processed)

    emails = [parse_email(r.encode()) for r in raws]
    data = train.SplitData(
        emails=emails[40:],
        target=target[40:],
        labels=np.where(target[40:] == 1, "phishing", "ham").astype(np.str_),
        clusters=[f"c{i}" for i in range(40, n)],
    )
    pipeline = lightgbm_pipeline(_SMOKE).fit(emails[:40], target[:40])
    model, _ = train.calibrate(pipeline, data)
    thresholds = train.operating_thresholds(model, data)
    train.track_run("lightgbm", model, _SMOKE, {}, processed, settings, thresholds)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bootstrap")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic model, for CI")
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")

    if has_production(settings):
        return 0
    if args.smoke:
        code = smoke(settings)
    else:
        steps = (download.main, dataset.main, train.main)
        code = next((c for step in steps if (c := step([])) != 0), 0)
    if code != 0:
        return code
    version = latest_version(settings.model_name, settings)
    promote(version, settings, settings.model_alias)
    logger.info("%s v%s promoted to @%s", version.name, version.version, settings.model_alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
