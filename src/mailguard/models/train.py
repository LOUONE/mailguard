"""Train the baselines on the training split and track every run in MLflow.

`train_baseline` takes the whole dataset and selects the training rows itself, so no caller can
fit a vectoriser on validation or test data by mistake. Each run records its full config, its
validation metrics, the git commit and the dataset hashes, and registers the fitted pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mailguard.config import Settings, get_settings
from mailguard.data.dataset import Split, load_dataset, load_manifest, manifest_hash
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, flatten, load_config
from mailguard.models.metrics import validation_metrics
from mailguard.models.pipelines import MODEL_NAMES, TRUSTED_TYPES, ModelName, build_pipeline
from mailguard.parsing import Email, parse_email

logger = logging.getLogger(__name__)

EXPERIMENT: Final = "mailguard-baselines"
REGISTERED_PREFIX: Final = "mailguard-"


@dataclass(frozen=True, slots=True)
class SplitData:
    emails: list[Email]
    target: NDArray[np.int64]
    labels: NDArray[np.str_]


def split_data(frame: pd.DataFrame, split: Split, column: str = "split") -> SplitData:
    rows = frame[frame[column] == split]
    return SplitData(
        emails=[parse_email(raw) for raw in rows["raw"]],
        target=rows["target"].to_numpy(dtype=np.int64),
        labels=rows["label"].to_numpy(dtype=np.str_),
    )


def train_baseline(
    name: ModelName, frame: pd.DataFrame, config: BaselineConfig, column: str = "split"
) -> Pipeline:
    """Fit one baseline on the training rows of `frame`, and on nothing else.

    `column` picks the assignment: `split` (main) or `temporal_split` (the temporal check).
    """
    train = split_data(frame, "train", column)
    logger.info("%s: fitting on %d training messages (%s)", name, len(train.emails), column)
    return build_pipeline(name, config).fit(train.emails, train.target)


def scores(pipeline: Pipeline, data: SplitData) -> NDArray[np.float64]:
    proba: NDArray[np.float64] = pipeline.predict_proba(data.emails)[:, 1]
    return proba


def git_state(cwd: Path | None = None) -> dict[str, str]:
    """Commit and dirty flag of the working tree, or "unknown" outside a git checkout."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain"))),
        }
    except OSError, subprocess.SubprocessError:
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def feature_summary(pipeline: Pipeline) -> dict[str, Any]:
    """Column count per feature group, and the names of the hand-made features.

    TF-IDF term names are left out: they are n-grams of corpus text.
    """
    groups: dict[str, Any] = {}
    for group, transformer in pipeline.named_steps["features"].transformer_list:
        names = [str(n) for n in transformer.get_feature_names_out()]
        groups[group] = {"columns": len(names)}
        if group != "text":
            groups[group]["names"] = names
    return groups


def runtime_requirements() -> list[str]:
    """Pinned versions of what a saved pipeline needs to load."""
    return [
        f"{name}=={version(name)}" for name in ("mailguard", "scikit-learn", "lightgbm", "numpy")
    ]


def model_params(name: ModelName, config: BaselineConfig) -> dict[str, Any]:
    section = config.logreg if name == "logreg" else config.lightgbm
    return {"seed": config.seed, **flatten(section.model_dump())}


def track_run(
    name: ModelName,
    pipeline: Pipeline,
    config: BaselineConfig,
    metrics: dict[str, float],
    processed_dir: Path,
    settings: Settings,
) -> str:
    """Log one fitted baseline to MLflow and register it. Returns the run id."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # slow import, only needed here
    import mlflow.sklearn

    if settings.mlflow_tracking_uri.startswith("sqlite:///"):
        Path(settings.mlflow_tracking_uri.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=settings.mlflow_artifact_dir.resolve().as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    manifest = load_manifest(processed_dir)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "model": name,
                "dataset_manifest_sha256": manifest_hash(processed_dir),
                "dataset_assignment_sha256": manifest["assignment_sha256"],
                **git_state(),
            }
        )
        mlflow.log_params(model_params(name, config))
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_dict(feature_summary(pipeline), "features.json")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=REGISTERED_PREFIX + name,
            serialization_format="skops",
            skops_trusted_types=list(TRUSTED_TYPES),
            # The pipeline only loads inside the mailguard environment; stating that is faster
            # and more honest than letting MLflow infer requirements in a subprocess.
            pip_requirements=runtime_requirements(),
        )
        run_id: str = run.info.run_id
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train", description="Train and track the baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument(
        "--no-tracking", action="store_true", help="fit and report without logging to MLflow"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(message)s")
    processed = settings.data_dir / "processed"
    config = load_config(args.config)
    try:
        frame = load_dataset(processed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    val = split_data(frame, "val")
    names: list[ModelName] = list(MODEL_NAMES) if args.model == "all" else [args.model]
    for name in names:
        pipeline = train_baseline(name, frame, config)
        metrics = validation_metrics(val.target, val.labels, scores(pipeline, val))
        logger.info(
            "%s: val PR-AUC %.4f, recall %.4f at 1%% ham FPR, %.4f at 0.1%%",
            name,
            metrics["pr_auc"],
            metrics["recall_at_1pct_fpr"],
            metrics["recall_at_01pct_fpr"],
        )
        if not args.no_tracking:
            run_id = track_run(name, pipeline, config, metrics, processed, settings)
            logger.info("%s: MLflow run %s", name, run_id)
    return 0
