"""Compare model variants over several split seeds, because one split cannot settle it.

The update to ADR 0006 is the reason this module exists: an effect of a couple of points
reversed when 850 messages changed split. The cluster bootstrap resamples a test set; it does
not show how much the answer depends on which clusters landed in train, validation and test.
Here each seed re-splits the corpus (same grouped, stratified procedure as `make dataset`),
every variant is trained on that split, and the comparison is repeated. A difference that
keeps its sign across seeds is a finding; one that flips is noise.

Per seed and variant, at a 1% FPR threshold chosen on that seed's validation ham, either all of
it or only the ham that is not list mail (the "personal threshold"):
recall, ham FPR split into mailing-list ham and other ham (#73), spam FPR, and the share of a
probe of made-up personal emails flagged (`mailguard.data.synthetic`).
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mailguard.data.dataset import SEED, add_splits
from mailguard.models.config import BaselineConfig
from mailguard.models.metrics import threshold_at_fpr
from mailguard.models.pipelines import lightgbm_pipeline
from mailguard.parsing import Email

logger = logging.getLogger(__name__)

FPR: Final = 0.01
METRICS: Final = ("recall", "ham_fpr", "list_ham_fpr", "other_ham_fpr", "spam_fpr", "probe_flagged")


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    config: BaselineConfig
    # Weight ham that is not list mail so that it counts as much as list ham in training.
    reweight_personal_ham: bool = False


def seeds(n: int) -> list[int]:
    """The reference seed first, then the next ones: seed 0 of every study is the served split."""
    return [SEED + i for i in range(n)]


def sample_weights(labels: NDArray[np.str_], is_list: NDArray[np.bool_]) -> NDArray[np.float64]:
    ham = labels == "ham"
    personal, listed = ham & ~is_list, ham & is_list
    weights = np.ones(len(labels))
    if personal.any() and listed.any():
        weights[personal] = listed.sum() / personal.sum()
    return weights


def rate(flags: NDArray[np.bool_], mask: NDArray[np.bool_]) -> float:
    return float(flags[mask].mean()) if mask.any() else float("nan")


def run_seed(
    frame: pd.DataFrame,
    emails: Sequence[Email],
    is_list: NDArray[np.bool_],
    probe: Sequence[Email],
    seed: int,
    variants: Sequence[Variant],
) -> list[dict[str, Any]]:
    """Every variant on one split. `emails` and `is_list` follow the rows of `frame`."""
    split = add_splits(frame.drop(columns=["split", "temporal_split"], errors="ignore"), seed=seed)
    split = split.loc[frame.index]
    labels = frame["label"].to_numpy(dtype=np.str_)
    target = frame["target"].to_numpy(dtype=np.int64)
    part = {
        name: np.flatnonzero(split["split"].to_numpy() == name) for name in ("train", "val", "test")
    }

    def subset(name: str) -> list[Email]:
        return [emails[int(i)] for i in part[name]]

    train, val, test = subset("train"), subset("val"), subset("test")
    val_ham = labels[part["val"]] == "ham"
    val_personal = val_ham & ~is_list[part["val"]]
    test_labels, test_list = labels[part["test"]], is_list[part["test"]]
    ham = test_labels == "ham"
    out = []
    for variant in variants:
        weights = (
            sample_weights(labels[part["train"]], is_list[part["train"]])
            if variant.reweight_personal_ham
            else None
        )
        pipeline = lightgbm_pipeline(variant.config)
        pipeline.fit(train, target[part["train"]], classifier__sample_weight=weights)
        val_scores = pipeline.predict_proba(val)[:, 1]
        test_scores = pipeline.predict_proba(test)[:, 1]
        probe_scores = pipeline.predict_proba(probe)[:, 1]
        # Two ways to set the same 1% target: on all validation ham, three quarters of which is
        # list mail that scores near zero, or on the ham that is not list mail, the kind a real
        # inbox is full of. Same model, different decision.
        rules = {
            variant.name: val_scores[val_ham],
            f"{variant.name}, personal threshold": val_scores[val_personal],
        }
        for name, reference in rules.items():
            threshold = threshold_at_fpr(reference, FPR)
            flags = test_scores > threshold
            result: dict[str, Any] = {
                "seed": seed,
                "variant": name,
                "threshold": threshold,
                "recall": rate(flags, test_labels == "phishing"),
                "ham_fpr": rate(flags, ham),
                "list_ham_fpr": rate(flags, ham & test_list),
                "other_ham_fpr": rate(flags, ham & ~test_list),
                "spam_fpr": rate(flags, test_labels == "spam"),
                "probe_flagged": float((probe_scores > threshold).mean()),
            }
            logger.info(
                "seed %d %s: %s",
                seed,
                name,
                {k: round(v, 4) for k, v in result.items() if k in METRICS},
            )
            out.append(result)
    return out


LABELS: Final = {
    "recall": "Recall (phishing caught)",
    "ham_fpr": "Ham FPR, all ham",
    "list_ham_fpr": "Ham FPR, mailing-list ham",
    "other_ham_fpr": "Ham FPR, other ham",
    "spam_fpr": "Spam flagged",
    "probe_flagged": "Made-up personal emails flagged",
}


def render(summary: dict[str, Any], title: str) -> str:
    """Markdown tables, one per metric: variants, their spread, and paired differences."""
    seed_count = len(summary["seeds"])
    lines = [
        f"# {title}",
        "",
        "Generated by `scripts/experiment_mailing_lists.py`. Metrics only. Each of the "
        f"{seed_count} seeds re-splits the corpus; every variant is trained and measured on each "
        "split, at the 1% ham-FPR threshold of that split's validation ham. Differences are "
        f"paired with `{summary['baseline']}` on the same split.",
        "",
    ]
    for metric in METRICS:
        lines += [
            f"## {LABELS[metric]}",
            "",
            "| Variant | Mean | Range over seeds | Difference, mean | Difference, range | "
            "Seeds lower / higher |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in summary["variants"].items():
            s = entry[metric]
            spread = f"{100 * s['min']:.2f} to {100 * s['max']:.2f}"
            row = f"| {name} | {100 * s['mean']:.2f}% | {spread} |"
            if "difference_mean" in s:
                row += (
                    f" {100 * s['difference_mean']:+.2f} | {100 * s['difference_min']:+.2f} to "
                    f"{100 * s['difference_max']:+.2f} | {s['seeds_lower']} / {s['seeds_higher']} |"
                )
            else:
                row += " | | |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def summarise(results: Sequence[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Mean and spread per variant, and the paired difference with the baseline, seed by seed."""
    by_seed = {(r["seed"], r["variant"]): r for r in results}
    variants = list(dict.fromkeys(r["variant"] for r in results))
    seed_list = sorted({r["seed"] for r in results})
    summary: dict[str, Any] = {"seeds": seed_list, "baseline": baseline, "variants": {}}
    for variant in variants:
        entry: dict[str, Any] = {}
        for metric in METRICS:
            values = [by_seed[(s, variant)][metric] for s in seed_list]
            stats = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
            if variant != baseline:
                diffs = [
                    v - by_seed[(s, baseline)][metric]
                    for s, v in zip(seed_list, values, strict=True)
                ]
                stats |= {
                    "difference_mean": statistics.fmean(diffs),
                    "difference_min": min(diffs),
                    "difference_max": max(diffs),
                    "seeds_lower": sum(d < 0 for d in diffs),
                    "seeds_higher": sum(d > 0 for d in diffs),
                }
            entry[metric] = stats
        summary["variants"][variant] = entry
    return summary
