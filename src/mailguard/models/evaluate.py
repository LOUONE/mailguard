"""Final evaluation of the baselines on the test split, with cluster-bootstrap intervals.

Thresholds are never chosen on test. Each operating point (1% and 0.1% ham FPR) is fixed on
validation ham and then applied to test as is, so the test FPR is a measurement, not a target.
Every number comes with a 95% interval from a bootstrap over duplicate clusters, and the two
baselines are compared on the same replicates (paired) and with McNemar's test.

The temporal check refits each baseline on phishing up to 2019 and measures it on 2020-2025
phishing, against the same baseline trained on the main split, on the same 2020-2025 messages.

The report also says how far the test set can be trusted: its resolution on ham, the slices
where the model is weak, and an ablation of the one feature suspected of encoding the corpus era.

The report holds metrics only (no message content, no run id or timestamp), so `make evaluate`
regenerates it identically and it can be committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from mailguard.models.calibration import (
    CalibratedPipeline,
    adjust_prior,
    expected_calibration_error,
    precision_at_prior,
    reliability_table,
)
from mailguard.models.config import DEFAULT_CONFIG, BaselineConfig, load_config
from mailguard.models.explain import contributions, explain, global_importance, group_importance
from mailguard.models.metrics import FPR_TARGETS, threshold_at_fpr
from mailguard.models.pipelines import MODEL_NAMES, ModelName
from mailguard.models.plots import reliability_svg
from mailguard.models.slices import slice_masks
from mailguard.models.stats import (
    ClusterBootstrap,
    mcnemar,
    weighted_pr_auc,
    weighted_rate,
)
from mailguard.models.train import EXPERIMENT, REGISTERED_PREFIX, train_baseline
from mailguard.parsing import Email, parse_email

logger = logging.getLogger(__name__)

N_BOOT: Final = 1000
BOOT_SEED: Final = 20261027
REPORT_DIR: Final = Path("reports")
RELIABILITY_SVG: Final = "reliability.svg"
# Phishing base rates of a real mailbox, for precision and probability conversions. The corpus
# itself is 57% phishing; a filtered inbox sees well under 1%.
PRODUCTION_PRIORS: Final = (0.001, 0.005, 0.01, 0.05)
PER_MESSAGES: Final = 1000
# The strongest LightGBM feature, and a probable era confound (docs/error_analysis.md): the
# report measures what the model loses without it.
ABLATED_FEATURE: Final = "url_max_length"

type Model = Pipeline | CalibratedPipeline

TOP_FEATURES: Final = 20
# A made-up message (RFC 2606 domains) whose local explanation goes in the report, so a reader
# sees what the API will return without any corpus content being published.
SYNTHETIC_EXAMPLE: Final = b"""\
From: "PayPal Security" <alerts@secure-paypal-verify.example>
Reply-To: help@other.example
Subject: Urgent: verify your account within 24 hours
Content-Type: text/html

<p>Dear customer, unusual activity was detected and your account is suspended.</p>
<p>Please <a href="http://198.51.100.23/paypal/login.php?session=8f3a9c2e">www.paypal.com</a>
to confirm your password and billing information.</p>
"""


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


def parse_rows(rows: Rows) -> list[Email]:
    """Emails in the row order of `rows`, so scores line up with its labels and clusters."""
    return [parse_email(raw) for raw in rows.frame["raw"]]


def predict(model: Model, rows: Rows | list[Email]) -> NDArray[np.float64]:
    emails = parse_rows(rows) if isinstance(rows, Rows) else rows
    scores: NDArray[np.float64] = model.predict_proba(emails)[:, 1]
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


def calibration_report(
    rows: Rows, model: CalibratedPipeline, emails: list[Email], boot: ClusterBootstrap
) -> dict[str, Any]:
    """ECE and Brier score of the raw and calibrated scores on test, with reliability bins."""
    target = rows.target
    raw, calibrated = model.raw_scores(emails), predict(model, emails)

    def brier(scores: NDArray[np.float64]) -> dict[str, float]:
        return boot.interval(
            lambda w: float(np.average((scores - target) ** 2, weights=w))
        ).as_dict()

    def ece(scores: NDArray[np.float64]) -> dict[str, float]:
        return boot.interval(lambda w: expected_calibration_error(target, scores, w)).as_dict()

    return {
        "method": model.calibrator.method,
        "fitted_on_prior": model.calibrator.prior_,
        "ece_raw": ece(raw),
        "ece_calibrated": ece(calibrated),
        "brier_raw": brier(raw),
        "brier_calibrated": brier(calibrated),
        "reliability_raw": reliability_table(target, raw),
        "reliability_calibrated": reliability_table(target, calibrated),
    }


def base_rate_report(
    rows: Rows,
    scores: NDArray[np.float64],
    threshold: float,
    fitted_on_prior: float,
    boot: ClusterBootstrap,
) -> dict[str, Any]:
    """What the 1% operating point and the probabilities mean in a mailbox with little phishing.

    Precision is computed from recall and ham FPR on test, which do not depend on the base rate,
    and carries their bootstrap uncertainty.
    """
    flagged = scores > threshold
    phishing, ham = rows.is_label("phishing"), rows.is_label("ham")

    def precision(prior: float) -> dict[str, float]:
        return boot.interval(
            lambda w: precision_at_prior(
                weighted_rate(flagged, phishing, w), weighted_rate(flagged, ham, w), prior
            )
        ).as_dict()

    def per_messages(prior: float) -> dict[str, dict[str, float]]:
        """Phishing caught and ham flagged among 1,000 messages with this share of phishing."""
        caught, false_alarms = PER_MESSAGES * prior, PER_MESSAGES * (1 - prior)
        return {
            "phishing_caught": boot.interval(
                lambda w: caught * weighted_rate(flagged, phishing, w)
            ).as_dict(),
            "ham_flagged": boot.interval(
                lambda w: false_alarms * weighted_rate(flagged, ham, w)
            ).as_dict(),
        }

    return {
        "precision_at_1pct_threshold": {str(p): precision(p) for p in PRODUCTION_PRIORS},
        "per_1000_messages": {str(p): per_messages(p) for p in PRODUCTION_PRIORS},
        "probability_at_prior": {
            str(prior): {
                str(p): float(adjust_prior(p, fitted_on_prior, prior)) for p in (0.5, 0.9, 0.99)
            }
            for prior in PRODUCTION_PRIORS
        },
    }


def explanation_report(model: CalibratedPipeline, emails: list[Email]) -> dict[str, Any]:
    """Global feature importance on test, by group and per feature, plus the synthetic example."""
    c = contributions(model.pipeline, emails)
    example = parse_email(SYNTHETIC_EXAMPLE)
    return {
        "group_share": group_importance(c),
        "top_features": [
            {"feature": name, "mean_abs_contribution": value}
            for name, value in global_importance(c)[:TOP_FEATURES]
        ],
        "example": {
            "probability": float(predict(model, [example])[0]),
            "reasons": explain(model.pipeline, example, top=5),
        },
    }


def resolution(val: Rows, test: Rows) -> dict[str, Any]:
    """What the ham of each split can measure.

    A threshold for an FPR target lets floor(target * n) validation ham through; at zero it is
    simply the highest validation ham score. On test, one ham message is 1 / n of the FPR, and
    showing that an FPR is below a target with 95% confidence takes about 3 / target ham messages
    with none flagged (the rule of three).
    """
    val_ham, test_ham = int(val.is_label("ham").sum()), int(test.is_label("ham").sum())
    return {
        "validation_ham": val_ham,
        "test_ham": test_ham,
        "one_test_ham": 1 / test_ham if test_ham else math.nan,
        "targets": {
            name: {
                "validation_ham_allowed": math.floor(fpr * val_ham + 1e-9),
                "test_ham_needed": math.ceil(3 / fpr - 1e-9),
            }
            for name, fpr in FPR_TARGETS.items()
        },
    }


def slice_report(
    rows: Rows,
    emails: list[Email],
    scores: Mapping[ModelName, NDArray[np.float64]],
    thresholds: Mapping[ModelName, Mapping[str, float]],
    boot: ClusterBootstrap,
) -> list[dict[str, Any]]:
    """Recall and ham FPR at the 1% operating point, in each slice of `mailguard.models.slices`.

    A list rather than a mapping: the JSON report is written with sorted keys, and slices must
    keep their order.
    """
    phishing, ham = rows.is_label("phishing"), rows.is_label("ham")
    out: list[dict[str, Any]] = []
    for name, mask in slice_masks(emails).items():
        positives, negatives = mask & phishing, mask & ham
        models: dict[str, Any] = {}
        for model, s in scores.items():
            flagged = s > thresholds[model]["1pct"]
            models[model] = {
                "recall": rate(boot, flagged, positives) if positives.any() else None,
                "ham_fpr": rate(boot, flagged, negatives) if negatives.any() else None,
            }
        out.append(
            {"slice": name, "phishing": int(positives.sum()), "ham": int(negatives.sum()), **models}
        )
    return out


def ablation_report(
    frame: pd.DataFrame,
    val: Rows,
    test: Rows,
    emails: tuple[list[Email], list[Email]],
    reference: tuple[NDArray[np.float64], float],
    ablated: tuple[Pipeline, Pipeline],
    boot: ClusterBootstrap,
    n_boot: int,
) -> dict[str, Any]:
    """LightGBM retrained without `ABLATED_FEATURE`, paired against the registered LightGBM.

    `emails` are the validation and test emails, `reference` the registered model's test scores
    and 1% threshold, `ablated` the pipelines trained without the feature on the main and on the
    temporal split. Differences are ablated minus reference.
    """
    pipeline, temporal_pipeline = ablated
    val_scores, test_scores = predict(pipeline, emails[0]), predict(pipeline, emails[1])
    thresholds = val_thresholds(val, val_scores)
    flag_ref, flag_new = reference[0] > reference[1], test_scores > thresholds["1pct"]
    target = test.target
    return {
        "feature": ABLATED_FEATURE,
        "model": model_report(test, test_scores, thresholds, boot),
        "difference": {
            "pr_auc": boot.interval(
                lambda w: (
                    weighted_pr_auc(target, test_scores, w)
                    - weighted_pr_auc(target, reference[0], w)
                )
            ).as_dict(),
            "recall": rate_difference(boot, flag_ref, flag_new, test.is_label("phishing")),
            "ham_fpr": rate_difference(boot, flag_ref, flag_new, test.is_label("ham")),
            "spam_fpr": rate_difference(boot, flag_ref, flag_new, test.is_label("spam")),
        },
        "temporal": temporal_report(
            frame, test, test_scores, thresholds["1pct"], temporal_pipeline, n_boot
        ),
    }


def evaluate(
    frame: pd.DataFrame,
    pipelines: Mapping[ModelName, Model],
    temporal_pipelines: Mapping[ModelName, Pipeline],
    n_boot: int = N_BOOT,
    ablated: tuple[Pipeline, Pipeline] | None = None,
) -> dict[str, Any]:
    """The whole report. `ablated`: LightGBM without `ABLATED_FEATURE`, main and temporal."""
    val, test = Rows.of(frame, "split", "val"), Rows.of(frame, "split", "test")
    val_emails, test_emails = parse_rows(val), parse_rows(test)
    val_scores = {n: predict(p, val_emails) for n, p in pipelines.items()}
    test_scores = {n: predict(p, test_emails) for n, p in pipelines.items()}
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
        "resolution": resolution(val, test),
        "models": {n: model_report(test, test_scores[n], thresholds[n], boot) for n in pipelines},
        "slices": slice_report(test, test_emails, test_scores, thresholds, boot),
    }
    if set(pipelines) == set(MODEL_NAMES):
        report["comparison"] = comparison(test, test_scores, thresholds, boot)
    calibrated = {n: m for n, m in pipelines.items() if isinstance(m, CalibratedPipeline)}
    report["calibration"] = {
        n: calibration_report(test, m, test_emails, boot) for n, m in calibrated.items()
    }
    report["base_rate"] = {
        n: base_rate_report(test, test_scores[n], thresholds[n]["1pct"], m.calibrator.prior_, boot)
        for n, m in calibrated.items()
    }
    report["explanations"] = {n: explanation_report(m, test_emails) for n, m in calibrated.items()}
    report["temporal"] = {
        n: temporal_report(
            frame, test, test_scores[n], thresholds[n]["1pct"], temporal_pipelines[n], n_boot
        )
        for n in temporal_pipelines
    }
    if ablated is not None and "lightgbm" in pipelines and "lightgbm" in temporal_pipelines:
        report["ablation"] = ablation_report(
            frame,
            val,
            test,
            (val_emails, test_emails),
            (test_scores["lightgbm"], thresholds["lightgbm"]["1pct"]),
            ablated,
            boot,
            n_boot,
        )
    return report


# --- report rendering ----------------------------------------------------------------------


def _pct(i: Mapping[str, float]) -> str:
    return f"{100 * i['estimate']:.1f}% [{100 * i['low']:.1f}, {100 * i['high']:.1f}]"


def _num(i: Mapping[str, float], digits: int = 4) -> str:
    return f"{i['estimate']:.{digits}f} [{i['low']:.{digits}f}, {i['high']:.{digits}f}]"


def _models(by_model: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Entries in `MODEL_NAMES` order, so a report read back from sorted JSON renders the same."""
    return [(name, by_model[name]) for name in MODEL_NAMES if name in by_model]


def _count(i: Mapping[str, float]) -> str:
    return _num(i, 1)


def _pct_or_na(i: Mapping[str, float] | None) -> str:
    return _pct(i) if i else "n/a"


def _point_heading(report: Mapping[str, Any], point: str, label: str) -> str:
    allowed = report["resolution"]["targets"][point]["validation_ham_allowed"]
    if allowed:
        return f"## Operating point: {label} ham FPR on validation"
    return f"## Operating point: {label} ham FPR on validation, i.e. no validation ham flagged"


def _resolution_markdown(report: Mapping[str, Any]) -> list[str]:
    r = report["resolution"]
    lines = [
        "## What this test set can measure",
        "",
        f"- One ham message is {100 * r['one_test_ham']:.2f}% of test ham: a ham FPR is a count "
        "of a few messages, and its interval is wide.",
    ]
    for point, label in (("1pct", "1%"), ("01pct", "0.1%")):
        t = r["targets"][point]
        lines.append(
            f"- {label} target: {t['validation_ham_allowed']} of {r['validation_ham']} validation "
            f"ham may be flagged when the threshold is set. Showing a test FPR below {label} with "
            f"95% confidence takes about {t['test_ham_needed']:,} test ham with none flagged "
            f"(rule of three); the test set holds {r['test_ham']}."
        )
    lines += [
        "- So the 0.1% operating point is a stricter threshold, not a measured 0.1% FPR.",
        "",
    ]
    return lines


def _slices_markdown(report: Mapping[str, Any]) -> list[str]:
    slices = report["slices"]
    models = [n for n in MODEL_NAMES if n in report["models"]]
    lines = [
        "## Slices: where the model is weak",
        "",
        "At the 1% operating point. Words are those of the text the model reads (digits, "
        "addresses and URLs removed); English is judged by the share of common English function "
        "words, below 10% counting as little English (`mailguard.models.slices`). The edges were "
        "set on the training split.",
        "",
        "| Slice | Phishing | "
        + " | ".join(f"Recall {n}" for n in models)
        + " | Ham | "
        + " | ".join(f"Ham FPR {n}" for n in models)
        + " |",
        "|---|---|" + "---|" * len(models) + "---|" + "---|" * len(models),
    ]
    for s in slices:
        name = s["slice"]
        recalls = [_pct_or_na(s[n]["recall"]) for n in models]
        fprs = [_pct_or_na(s[n]["ham_fpr"]) for n in models]
        lines.append(
            f"| {name} | {s['phishing']} | "
            + " | ".join(recalls)
            + f" | {s['ham']} | "
            + " | ".join(fprs)
            + " |"
        )
    lines.append("")
    return lines


def _ablation_markdown(report: Mapping[str, Any]) -> list[str]:
    if "ablation" not in report:
        return []
    a = report["ablation"]
    feature, d = a["feature"], a["difference"]

    def row(name: str, model: Mapping[str, Any], temporal: Mapping[str, Any]) -> str:
        p = model["operating_points"]["1pct"]
        past = temporal["trained_up_to_2019"]
        return (
            f"| {name} | {_pct(p['recall'])} | {_pct(p['ham_fpr'])} ({p['ham_false_positives']} "
            f"msg) | {_pct(p['spam_fpr'])} | {_num(model['pr_auc'])} | "
            f"{_pct(past['recall_2020_2025'])}, ham FPR {_pct(past['ham_fpr'])} |"
        )

    return [
        f"## Ablation: LightGBM without `{feature}`",
        "",
        f"`{feature}` is the strongest LightGBM feature and a suspected era marker (short URLs in "
        "2002 ham). The same model is retrained without it, with the same configuration "
        "otherwise. Thresholds are set on validation as above.",
        "",
        "| Model | Recall at 1% | Ham FPR | Spam FPR | PR-AUC | Trained up to 2019: 2020-2025 "
        "recall |",
        "|---|---|---|---|---|---|",
        row("lightgbm", report["models"]["lightgbm"], report["temporal"]["lightgbm"]),
        row(f"lightgbm without `{feature}`", a["model"], a["temporal"]),
        "",
        f"Without minus with, paired: recall {_pct(d['recall'])}, ham FPR {_pct(d['ham_fpr'])}, "
        f"spam FPR {_pct(d['spam_fpr'])}, PR-AUC {_num(d['pr_auc'])}.",
        "",
    ]


def _explanation_markdown(report: Mapping[str, Any]) -> list[str]:
    explanations = report.get("explanations", {})
    if not explanations:
        return []
    groups = sorted({g for e in explanations.values() for g in e["group_share"]})
    lines = [
        "## What drives the scores",
        "",
        "Mean absolute contribution to the log-odds on test: exact TreeSHAP values for LightGBM, "
        "coefficient times value for the logistic regression (`mailguard.models.explain`). "
        "Share of the total by feature group:",
        "",
        "| Model | " + " | ".join(groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for name, e in _models(explanations):
        cells = [f"{100 * e['group_share'].get(g, 0.0):.1f}%" for g in groups]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name, e in _models(explanations):
        lines += ["", f"Top {len(e['top_features'])} features, {name}:", ""]
        lines += ["| # | Feature | Mean abs. contribution |", "|---|---|---|"]
        lines += [
            f"| {i} | `{f['feature']}` | {f['mean_abs_contribution']:.3f} |"
            for i, f in enumerate(e["top_features"], start=1)
        ]
    lines += [
        "",
        "Local explanation of a synthetic phishing email (`SYNTHETIC_EXAMPLE` in "
        "`mailguard.models.evaluate`), as the API will return it:",
        "",
        "| Model | Probability | Top reasons (log-odds) |",
        "|---|---|---|",
    ]
    for name, e in _models(explanations):
        reasons = ", ".join(
            f"`{r['feature']}` {r['contribution']:+.2f}" for r in e["example"]["reasons"]
        )
        lines.append(f"| {name} | {e['example']['probability']:.3f} | {reasons} |")
    lines.append("")
    return lines


def _calibration_markdown(report: Mapping[str, Any]) -> list[str]:
    calibration, base_rate = report.get("calibration", {}), report.get("base_rate", {})
    if not calibration:
        return []
    lines = [
        "## Calibration",
        "",
        "Calibrators are fitted on validation (method chosen by cross-validation grouped by "
        "cluster, on validation only) and measured here on test. ECE: mean gap between predicted "
        "and observed phishing rate over 10 equal-width bins.",
        "",
        "| Model | Method | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in _models(calibration):
        lines.append(
            f"| {name} | {c['method']} | {_pct(c['ece_raw'])} | {_pct(c['ece_calibrated'])} | "
            f"{_num(c['brier_raw'])} | {_num(c['brier_calibrated'])} |"
        )
    lines += [
        "",
        f"![Reliability diagram]({RELIABILITY_SVG})",
        "",
        "Almost every score is below 0.1 or above 0.9. The bins in between hold a handful of "
        "messages each, so their zigzag is noise; the counts are in the table below.",
        "",
        "| Model | Bin | Messages | Mean probability | Observed phishing rate |",
        "|---|---|---|---|---|",
    ]
    for name, c in _models(calibration):
        lines += [
            f"| {name} | {r['low']:.1f}-{r['high']:.1f} | {r['messages']} | "
            f"{r['mean_score']:.3f} | {r['phishing_rate']:.3f} |"
            for r in c["reliability_calibrated"]
        ]
    priors = [str(p) for p in PRODUCTION_PRIORS]
    lines += [
        "",
        "## What the numbers mean in a real mailbox",
        "",
        "The corpus is 57% phishing. Recall and FPR do not depend on that; precision and "
        "probabilities do. At the 1% operating point, out of 1,000 messages of which this share "
        "is phishing and the rest is ham:",
        "",
        "| Model | Phishing share | Phishing messages | Caught | Ham flagged (false alarms) "
        "| Precision |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in _models(base_rate):
        for p in priors:
            counts = b["per_1000_messages"][p]
            lines.append(
                f"| {name} | {100 * float(p):g}% | {PER_MESSAGES * float(p):g} | "
                f"{_count(counts['phishing_caught'])} | {_count(counts['ham_flagged'])} | "
                f"{_pct(b['precision_at_1pct_threshold'][p])} |"
            )
    lines += [
        "",
        "False alarms are ham FPR times the ham messages, so they carry its uncertainty: about "
        "five flagged test ham. An upper precision bound of 100% comes from replicates that drew "
        "no flagged ham at all. Spam is left out: flagging spam as phishing is the cheaper error "
        "(ADR 0001), and how much spam reaches a filtered inbox varies too much to assume.",
        "",
        "A calibrated probability converted to those base rates (Bayes' rule on the odds, "
        "`mailguard.models.calibration.adjust_prior`):",
        "",
        "| Model | Corpus probability | "
        + " | ".join(f"{100 * float(p):g}%" for p in priors)
        + " |",
        "|---|---|" + "---|" * len(priors),
    ]
    for name, b in _models(base_rate):
        for prob in ("0.5", "0.9", "0.99"):
            cells = [f"{b['probability_at_prior'][p][prob]:.3f}" for p in priors]
            lines.append(f"| {name} | {prob} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


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
        *_resolution_markdown(report),
    ]
    for point, label in (("1pct", "1%"), ("01pct", "0.1%")):
        lines += [
            _point_heading(report, point, label),
            "",
            "| Model | Recall | Recall 2004-2007 | Recall 2015-2025 | Ham FPR | Spam FPR |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in _models(report["models"]):
            p = m["operating_points"][point]
            lines.append(
                f"| {name} | {_pct(p['recall'])} | {_pct(p['recall_2004_2007'])} | "
                f"{_pct(p['recall_2015_2025'])} | {_pct(p['ham_fpr'])} "
                f"({p['ham_false_positives']} msg) | {_pct(p['spam_fpr'])} |"
            )
        lines.append("")
    lines += ["## Ranking", "", "| Model | PR-AUC |", "|---|---|"]
    lines += [f"| {n} | {_num(m['pr_auc'])} |" for n, m in _models(report["models"])]
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
    lines += _calibration_markdown(report)
    lines += _explanation_markdown(report)
    if report.get("temporal"):
        lines += [
            "## Temporal check: phishing from 2020-2025",
            "",
            "| Model | Trained up to 2019 | Trained on the main split |",
            "|---|---|---|",
        ]
        for name, t in _models(report["temporal"]):
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
    lines += _ablation_markdown(report)
    lines += _slices_markdown(report)
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Write the JSON, the Markdown and, when there is calibration data, the reliability SVG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "evaluation.json", out_dir / "evaluation.md"
    report_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    json_path.write_text(report_json, encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    paths = [json_path, md_path]
    if report.get("calibration"):
        svg_path = out_dir / RELIABILITY_SVG
        # Fixed order, so each model keeps its colour whatever order the report is read back in.
        calibration = report["calibration"]
        curves: dict[str, Any] = {
            n: calibration[n]["reliability_calibrated"] for n in MODEL_NAMES if n in calibration
        }
        svg_path.write_text(reliability_svg(curves), encoding="utf-8", newline="\n")
        paths.append(svg_path)
    return paths


# --- models from the registry --------------------------------------------------------------


def ablated_config(config: BaselineConfig) -> BaselineConfig:
    """`config` with `ABLATED_FEATURE` left out of LightGBM."""
    lightgbm = config.lightgbm
    exclude = (*lightgbm.exclude_features, ABLATED_FEATURE)
    return config.model_copy(
        update={"lightgbm": lightgbm.model_copy(update={"exclude_features": exclude})}
    )


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
    without = ablated_config(config)
    ablated = (
        train_baseline("lightgbm", frame, without),
        train_baseline("lightgbm", frame, without, "temporal_split"),
    )

    report = evaluate(frame, pipelines, temporal, args.n_boot, ablated)
    paths = write_report(report, args.out_dir)
    if not args.no_tracking:
        log_report(paths, settings)
    logger.info("report written to %s", ", ".join(map(str, paths)))
    return 0
