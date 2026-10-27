"""Final evaluation of the baselines on the test split, with cluster-bootstrap intervals.

Thresholds are never chosen on test. Each operating point (1% and 0.1% ham FPR) is fixed on
validation ham and then applied to test as is, so the test FPR is a measurement, not a target.
Every number comes with a 95% interval from a bootstrap over duplicate clusters, and the two
baselines are compared on the same replicates (paired) and with McNemar's test.

The temporal check refits each baseline on phishing up to 2019 and measures it on 2020-2025
phishing, against the same baseline trained on the main split, on the same 2020-2025 messages.

The report holds metrics only (no message content, no run id or timestamp), so `make evaluate`
regenerates it identically and it can be committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mailguard.config import Settings, get_settings
from mailguard.data.dataset import load_dataset, load_manifest
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, load_config
from mailguard.models.metrics import FPR_TARGETS, threshold_at_fpr
from mailguard.models.pipelines import MODEL_NAMES, ModelName
from mailguard.models.stats import (
    ClusterBootstrap,
    mcnemar,
    weighted_pr_auc,
    weighted_rate,
)
from mailguard.models.train import EXPERIMENT, REGISTERED_PREFIX, train_baseline
from mailguard.parsing import parse_email

logger = logging.getLogger(__name__)

N_BOOT: Final = 1000
BOOT_SEED: Final = 20261027
REPORT_DIR: Final = Path("reports")


@dataclass(frozen=True, slots=True)
class Rows:
    """The rows of one evaluation set, with everything metrics and the bootstrap need."""

    frame: pd.DataFrame

    @classmethod
    def of(cls, frame: pd.DataFrame, column: str, split: str) -> Rows:
        return cls(frame[frame[column] == split].reset_index(drop=True))

    @property
    def target(self) -> NDArray[np.int64]:
        return self.frame["target"].to_numpy(dtype=np.int64)

    def is_label(self, label: str) -> NDArray[np.bool_]:
        return self.frame["label"].eq(label).to_numpy(dtype=bool)

    def is_period(self, period: str) -> NDArray[np.bool_]:
        return self.is_label("phishing") & self.frame["period"].eq(period).to_numpy(dtype=bool)

    def is_recent(self, after: int) -> NDArray[np.bool_]:
        years = self.frame["year"].fillna(0).astype(int).to_numpy()
        return self.is_label("phishing") & (years > after)


def predict(pipeline: Pipeline, rows: Rows) -> NDArray[np.float64]:
    """Scores in the row order of `rows`, so they line up with its labels and clusters."""
    emails = [parse_email(raw) for raw in rows.frame["raw"]]
    scores: NDArray[np.float64] = pipeline.predict_proba(emails)[:, 1]
    return scores


def val_thresholds(val: Rows, scores: NDArray[np.float64]) -> dict[str, float]:
    return {
        name: threshold_at_fpr(scores[val.is_label("ham")], fpr)
        for name, fpr in FPR_TARGETS.items()
    }


Flags = NDArray[np.bool_]


def rate(boot: ClusterBootstrap, flagged: Flags, mask: Flags) -> dict[str, float]:
    """Share of `mask` rows that are flagged, with its interval."""
    return boot.interval(lambda w: weighted_rate(flagged, mask, w)).as_dict()


def rate_difference(
    boot: ClusterBootstrap, flagged_a: Flags, flagged_b: Flags, mask: Flags
) -> dict[str, float]:
    """Paired difference b - a of the flagged share of `mask` rows, with its interval."""
    return boot.interval(
        lambda w: weighted_rate(flagged_b, mask, w) - weighted_rate(flagged_a, mask, w)
    ).as_dict()


def operating_metrics(
    rows: Rows, scores: NDArray[np.float64], threshold: float, boot: ClusterBootstrap
) -> dict[str, Any]:
    flagged = scores > threshold
    masks = {
        "recall": rows.is_label("phishing"),
        "recall_2004_2007": rows.is_period("2004-2007"),
        "recall_2015_2025": rows.is_period("2015-2025"),
        "ham_fpr": rows.is_label("ham"),
        "spam_fpr": rows.is_label("spam"),
    }
    out: dict[str, Any] = {"threshold": threshold}
    out |= {name: rate(boot, flagged, mask) for name, mask in masks.items()}
    out["ham_false_positives"] = int((flagged & masks["ham_fpr"]).sum())
    return out


def model_report(
    rows: Rows, scores: NDArray[np.float64], thresholds: Mapping[str, float], boot: ClusterBootstrap
) -> dict[str, Any]:
    target = rows.target
    return {
        "pr_auc": boot.interval(lambda w: weighted_pr_auc(target, scores, w)).as_dict(),
        "operating_points": {
            name: operating_metrics(rows, scores, t, boot) for name, t in thresholds.items()
        },
    }


def comparison(
    rows: Rows,
    scores: Mapping[ModelName, NDArray[np.float64]],
    thresholds: Mapping[ModelName, Mapping[str, float]],
    boot: ClusterBootstrap,
    a: ModelName = "logreg",
    b: ModelName = "lightgbm",
) -> dict[str, Any]:
    """`b` minus `a`, at each operating point, on the same bootstrap replicates."""
    target = rows.target
    truth = target.astype(bool)
    phishing, ham = rows.is_label("phishing"), rows.is_label("ham")
    out: dict[str, Any] = {
        "pr_auc_difference": boot.interval(
            lambda w: weighted_pr_auc(target, scores[b], w) - weighted_pr_auc(target, scores[a], w)
        ).as_dict()
    }
    for name in FPR_TARGETS:
        flag_a = scores[a] > thresholds[a][name]
        flag_b = scores[b] > thresholds[b][name]
        out[name] = {
            "recall_difference": rate_difference(boot, flag_a, flag_b, phishing),
            "ham_fpr_difference": rate_difference(boot, flag_a, flag_b, ham),
            "mcnemar": mcnemar(flag_a == truth, flag_b == truth).as_dict(),
        }
    return out


def temporal_report(
    frame: pd.DataFrame,
    main_test: Rows,
    main_scores: NDArray[np.float64],
    main_threshold: float,
    temporal_pipeline: Pipeline,
    n_boot: int,
) -> dict[str, Any]:
    """Recall on 2020-2025 phishing: trained up to 2019 vs trained on the main split."""
    val = Rows.of(frame, "temporal_split", "val")
    test = Rows.of(frame, "temporal_split", "test")
    val_scores = predict(temporal_pipeline, val)
    test_scores = predict(temporal_pipeline, test)
    threshold = threshold_at_fpr(val_scores[val.is_label("ham")], FPR_TARGETS["1pct"])
    boot = ClusterBootstrap(test.frame["cluster"].tolist(), n_boot, BOOT_SEED)
    flagged = test_scores > threshold

    # The main-split model, on the 2020-2025 phishing that landed in the main test split.
    recent = main_test.is_recent(2019)
    main_boot = ClusterBootstrap(main_test.frame["cluster"].tolist(), n_boot, BOOT_SEED)
    return {
        "threshold": threshold,
        "trained_up_to_2019": {
            "recall_2020_2025": rate(boot, flagged, test.is_label("phishing")),
            "ham_fpr": rate(boot, flagged, test.is_label("ham")),
            "phishing_messages": int(test.is_label("phishing").sum()),
        },
        "trained_on_main_split": {
            "recall_2020_2025": rate(main_boot, main_scores > main_threshold, recent),
            "phishing_messages": int(recent.sum()),
        },
    }


def evaluate(
    frame: pd.DataFrame,
    pipelines: Mapping[ModelName, Pipeline],
    temporal_pipelines: Mapping[ModelName, Pipeline],
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    val, test = Rows.of(frame, "split", "val"), Rows.of(frame, "split", "test")
    val_scores = {n: predict(p, val) for n, p in pipelines.items()}
    test_scores = {n: predict(p, test) for n, p in pipelines.items()}
    thresholds = {n: val_thresholds(val, s) for n, s in val_scores.items()}
    boot = ClusterBootstrap(test.frame["cluster"].tolist(), n_boot, BOOT_SEED)

    counts = test.frame.groupby("label").size()
    report: dict[str, Any] = {
        "test_set": {
            "messages": len(test.frame),
            "clusters": int(test.frame["cluster"].nunique()),
            **{str(k): int(v) for k, v in counts.items()},
        },
        "bootstrap": {"replicates": n_boot, "unit": "duplicate cluster", "level": 0.95},
        "models": {n: model_report(test, test_scores[n], thresholds[n], boot) for n in pipelines},
    }
    if set(pipelines) == set(MODEL_NAMES):
        report["comparison"] = comparison(test, test_scores, thresholds, boot)
    report["temporal"] = {
        n: temporal_report(
            frame, test, test_scores[n], thresholds[n]["1pct"], temporal_pipelines[n], n_boot
        )
        for n in temporal_pipelines
    }
    return report


# --- report rendering ----------------------------------------------------------------------


def _pct(i: Mapping[str, float]) -> str:
    return f"{100 * i['estimate']:.1f}% [{100 * i['low']:.1f}, {100 * i['high']:.1f}]"


def _num(i: Mapping[str, float], digits: int = 4) -> str:
    return f"{i['estimate']:.{digits}f} [{i['low']:.{digits}f}, {i['high']:.{digits}f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    test = report["test_set"]
    lines = [
        "# Evaluation report",
        "",
        "Generated by `make evaluate`. Do not edit by hand. Metrics only, no message content.",
        "",
        f"Test split: {test['messages']:,} messages ({test.get('ham', 0)} ham, "
        f"{test.get('spam', 0)} spam, {test.get('phishing', 0)} phishing) in "
        f"{test['clusters']:,} duplicate clusters. Brackets are 95% intervals from "
        f"{report['bootstrap']['replicates']} bootstrap replicates over clusters. Thresholds are "
        "fixed on validation ham, so the test FPR is measured, not chosen.",
        "",
    ]
    for point, label in (("1pct", "1%"), ("01pct", "0.1%")):
        lines += [
            f"## Operating point: {label} ham FPR on validation",
            "",
            "| Model | Recall | Recall 2004-2007 | Recall 2015-2025 | Ham FPR | Spam FPR |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in report["models"].items():
            p = m["operating_points"][point]
            lines.append(
                f"| {name} | {_pct(p['recall'])} | {_pct(p['recall_2004_2007'])} | "
                f"{_pct(p['recall_2015_2025'])} | {_pct(p['ham_fpr'])} "
                f"({p['ham_false_positives']} msg) | {_pct(p['spam_fpr'])} |"
            )
        lines.append("")
    lines += ["## Ranking", "", "| Model | PR-AUC |", "|---|---|"]
    lines += [f"| {n} | {_num(m['pr_auc'])} |" for n, m in report["models"].items()]
    lines.append("")
    if "comparison" in report:
        c = report["comparison"]
        lines += [
            "## LightGBM minus logistic regression (paired)",
            "",
            f"PR-AUC difference: {_num(c['pr_auc_difference'])}.",
            "",
            "| Operating point | Recall difference | Ham FPR difference "
            "| McNemar: only LR / only LGBM correct | p |",
            "|---|---|---|---|---|",
        ]
        for point, label in (("1pct", "1%"), ("01pct", "0.1%")):
            m = c[point]["mcnemar"]
            lines.append(
                f"| {label} | {_pct(c[point]['recall_difference'])} | "
                f"{_pct(c[point]['ham_fpr_difference'])} | "
                f"{m['only_a_correct']} / {m['only_b_correct']} | {m['p_value']:.2g} |"
            )
        lines += [
            "",
            "McNemar assumes independent messages; duplicates make its p-value optimistic, so the "
            "paired cluster-bootstrap interval of the difference is the number to trust.",
            "",
        ]
    if report.get("temporal"):
        lines += [
            "## Temporal check: phishing from 2020-2025",
            "",
            "| Model | Trained up to 2019 | Trained on the main split |",
            "|---|---|---|",
        ]
        for name, t in report["temporal"].items():
            past, main = t["trained_up_to_2019"], t["trained_on_main_split"]
            lines.append(
                f"| {name} | {_pct(past['recall_2020_2025'])} on {past['phishing_messages']} msg, "
                f"ham FPR {_pct(past['ham_fpr'])} | "
                f"{_pct(main['recall_2020_2025'])} on {main['phishing_messages']} msg |"
            )
        lines += [
            "",
            "Both at a 1% ham FPR threshold fixed on validation. The gap is what a model loses "
            "on campaigns from years it has never seen.",
            "",
        ]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "evaluation.json", out_dir / "evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


# --- models from the registry --------------------------------------------------------------


def load_registered(name: ModelName, settings: Settings, assignment: str) -> Pipeline:
    """Latest registered version of a baseline, refusing one trained on another split."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    import mlflow.sklearn

    client = mlflow.MlflowClient(settings.mlflow_tracking_uri)
    versions = client.search_model_versions(f"name = '{REGISTERED_PREFIX}{name}'")
    if not versions:
        raise LookupError(f"no registered {name} model: run `make train` first")
    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id) if latest.run_id else None
    trained_on = run.data.tags.get("dataset_assignment_sha256") if run else None
    if trained_on != assignment:
        raise LookupError(
            f"{name} v{latest.version} was trained on another split: rerun `make train`"
        )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    pipeline: Pipeline = mlflow.sklearn.load_model(f"models:/{latest.name}/{latest.version}")
    return pipeline


def log_report(paths: Sequence[Path], settings: Settings) -> None:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluation"):
        for path in paths:
            mlflow.log_artifact(str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate", description="Evaluate the baselines on test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--no-tracking", action="store_true", help="do not log the report")
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(message)s")
    processed = settings.data_dir / "processed"
    config: BaselineConfig = load_config(args.config)
    try:
        frame = load_dataset(processed)
        assignment = load_manifest(processed)["assignment_sha256"]
        pipelines = {n: load_registered(n, settings, assignment) for n in MODEL_NAMES}
    except (FileNotFoundError, LookupError) as exc:
        logger.error("%s", exc)
        return 1
    temporal = {n: train_baseline(n, frame, config, "temporal_split") for n in MODEL_NAMES}

    report = evaluate(frame, pipelines, temporal, args.n_boot)
    paths = write_report(report, args.out_dir)
    if not args.no_tracking:
        log_report(paths, settings)
    logger.info("report written to %s", ", ".join(map(str, paths)))
    return 0
