"""Natural drift, end to end: a model that never saw 2024-2025 phishing meets it in traffic.

    uv run python scripts/simulate_drift.py        # about 10 minutes, needs make dataset

The corpora already contain the drift a production system faces: campaigns from years a model
has not seen. Instead of synthetic noise, this scenario holds out every 2024-2025 phishing
message (and the duplicate clusters they belong to) and replays the story:

1. **The old world.** Train LightGBM without the held-out campaigns and put it in production.
2. **Normal traffic.** Replay in-distribution messages through the API: the drift job should be
   quiet.
3. **The new campaigns arrive.** Replay the 2024-2025 phishing, mixed with ordinary ham and spam:
   does the drift job notice, and on which features?
4. **Retraining.** Run the retraining flow on the full corpus: is the candidate, which has seen
   the new campaigns, significantly better than production, by the rule of ADR 0008?

Everything happens in its own folder (`data/simulation/`), with its own MLflow: the real registry
and the model in production are never touched. The output is numbers only.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from mailguard.config import Settings, get_settings
from mailguard.data.dataset import assignment_hash, load_dataset, write_dataset
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, load_config
from mailguard.models.registry import promote
from mailguard.monitoring import drift_job
from mailguard.pipeline import retrain
from mailguard.serving.app import create_app
from mailguard.serving.model import ServedModel, load_production

logger = logging.getLogger(__name__)

HELD_OUT_FROM: Final = 2024
OUT: Final = Path("reports/drift_simulation.json")


def held_out(frame: pd.DataFrame, from_year: int = HELD_OUT_FROM) -> pd.Series:
    """Phishing from `from_year` on, plus every message in the same duplicate clusters."""
    years = frame["year"].fillna(0).astype(int)
    recent = frame["label"].eq("phishing") & (years >= from_year)
    return frame["cluster"].isin(set(frame.loc[recent, "cluster"]))


def workspace(name: str, root: Path) -> Settings:
    """Settings of one phase: shared data and MLflow under `root`, its own traffic and drift.

    WARNING level: replaying thousands of requests would otherwise print as many access lines.
    """
    return Settings(
        data_dir=root / "data",
        mlflow_tracking_uri=f"sqlite:///{(root / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_artifact_dir=root / "mlruns" / "artifacts",
        feature_log_dir=root / "traffic" / name,
        drift_dir=root / "drift" / name,
        log_level="WARNING",
    )


def write(frame: pd.DataFrame, settings: Settings) -> None:
    manifest = {"assignment_sha256": assignment_hash(frame), "rows": len(frame)}
    write_dataset(frame, manifest, settings.data_dir / "processed")


def replay(model: ServedModel, settings: Settings, raws: Sequence[bytes]) -> dict[str, Any]:
    """Send `raws` through the real API code path, then run the drift job once."""
    import base64

    app = create_app(settings, loader=lambda _: model)
    labels = []
    with TestClient(app) as client:
        for raw in raws:
            body = {"raw": base64.b64encode(raw).decode(), "encoding": "base64"}
            labels.append(client.post("/v1/predict", json=body).json()["label"])
    summary = drift_job.run_once(settings, model, window=len(raws))
    if summary is None:
        raise RuntimeError("too little traffic for a drift report")
    top = [
        {k: f[k] for k in ("feature", "psi", "reference_mean", "current_mean")}
        for f in summary["features"][:8]
    ]
    return {
        "messages": len(raws),
        "flagged_phishing": labels.count("phishing"),
        "drifted": summary["drifted"],
        "share_drifted": summary["share_drifted"],
        "score_drifted": summary["score_drifted"],
        "top_features": top,
    }


def recall_on(model: ServedModel, raws: Sequence[bytes]) -> float:
    from mailguard.parsing import parse_email

    flags = [model.predict(parse_email(r), 0.01).is_phishing for r in raws]
    return float(np.mean(flags))


def simulate(
    frame: pd.DataFrame,
    config: BaselineConfig,
    root: Path,
    seed: int = 0,
    from_year: int = HELD_OUT_FROM,
) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    held = held_out(frame, from_year)
    old_world, new_campaigns = frame[~held], frame[held & frame["label"].eq("phishing")]
    logger.info("held out %d messages (%d phishing)", int(held.sum()), len(new_campaigns))

    # 1. The old world: train and promote a model that never saw the new campaigns.
    train_settings = workspace("train", root)
    write(old_world.reset_index(drop=True), train_settings)
    old = retrain.train_candidate(old_world.reset_index(drop=True), config, train_settings)
    promote(old, train_settings)
    production = load_production(train_settings)

    rng = np.random.default_rng(seed)
    test = old_world[old_world["split"] == "test"]
    legitimate = test[test["label"] != "phishing"]["raw"].tolist()
    old_phishing = test[test["label"] == "phishing"]["raw"].tolist()

    # 2. Normal traffic, then 3. the new campaigns mixed with the same legitimate mail.
    normal = replay(production, workspace("normal", root), legitimate + old_phishing)
    wave = list(new_campaigns["raw"]) + legitimate
    rng.shuffle(wave)
    arrival = replay(production, workspace("new-campaigns", root), wave)

    # 4. Retraining on the full corpus, compared with production by the rule of ADR 0008.
    write(frame.reset_index(drop=True), train_settings)
    outcome = retrain.retrain(train_settings, config)
    candidate = load_production(train_settings) if outcome.decision.promote else None
    new_raws = list(new_campaigns[new_campaigns["split"] == "test"]["raw"])
    return {
        "held_out": {"messages": int(held.sum()), "phishing": len(new_campaigns)},
        "production_version": production.version,
        "recall_on_new_campaigns": {
            "production": recall_on(production, list(new_campaigns["raw"])),
            "candidate_on_test_part": (
                recall_on(candidate, new_raws) if candidate is not None else None
            ),
        },
        "normal_traffic": normal,
        "new_campaigns": arrival,
        "retraining": outcome.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simulate_drift", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path("data/simulation"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")
    try:
        frame = load_dataset(settings.data_dir / "processed")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    result = simulate(frame, load_config(args.config), args.root)
    result.get("retraining", {}).pop("time", None)  # keep the committed file reproducible
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    logger.info("results in %s", args.out)
    return 0
