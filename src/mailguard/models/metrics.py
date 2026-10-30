"""Metrics that do not depend on the 57% phishing prior of the corpus (ADR 0001).

The headline is recall at a fixed false-positive rate measured on ham. A message is flagged when
its score is strictly above the threshold, and the threshold is chosen on held-out ham so that at
most `max_fpr` of it would be flagged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

FPR_TARGETS: Final[Mapping[str, float]] = {"1pct": 0.01, "01pct": 0.001}


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    recall: float  # share of phishing flagged
    ham_fpr: float  # share of ham flagged
    spam_fpr: float  # share of spam flagged, reported apart (ADR 0001)


def threshold_at_fpr(
    negative_scores: ArrayLike, max_fpr: float, groups: Sequence[str] | None = None
) -> float:
    """Lowest threshold that flags at most `max_fpr` of `negative_scores` (flag: score > t).

    With n negatives, at most k = floor(max_fpr * n) may be flagged, so the threshold is the
    (k + 1)-th highest negative score. Ties at the threshold are not flagged, which errs on the
    side of fewer false positives.

    With `groups` (duplicate clusters), each cluster counts once, by its highest score: k of the
    n clusters may be flagged. Nine copies of one email then weigh as one campaign, as they do in
    the splits and the bootstrap, instead of filling the top of the ranking by themselves.
    """
    values: NDArray[np.floating[Any]] = np.asarray(negative_scores, dtype=np.float64)
    if groups is not None:
        values = pd.Series(values).groupby(list(groups)).max().to_numpy(dtype=np.float64)
    scores = np.sort(values)[::-1]
    if scores.size == 0:
        raise ValueError("need at least one negative score")
    if not 0 <= max_fpr < 1:
        raise ValueError(f"max_fpr must be in [0, 1), got {max_fpr}")
    k = math.floor(max_fpr * scores.size + 1e-9)
    return float(scores[k])


def operating_point(
    scores: NDArray[np.float64], labels: NDArray[np.str_], threshold: float
) -> OperatingPoint:
    flagged = scores > threshold

    def rate(label: str) -> float:
        mask = labels == label
        return float(flagged[mask].mean()) if mask.any() else math.nan

    return OperatingPoint(threshold, rate("phishing"), rate("ham"), rate("spam"))


def ranking_metrics(target: NDArray[np.int64], scores: NDArray[np.float64]) -> dict[str, float]:
    """Threshold-free metrics on phishing against ham and spam."""
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return {
        "pr_auc": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(target, scores)),
    }


def validation_metrics(
    target: NDArray[np.int64], labels: NDArray[np.str_], scores: NDArray[np.float64]
) -> dict[str, float]:
    """Ranking metrics plus, for each FPR target, the threshold chosen on this set's ham.

    On validation this is what the operating thresholds are fitted on. The test set reuses those
    thresholds instead of choosing its own (#27).
    """
    metrics = ranking_metrics(target, scores)
    ham_scores = scores[labels == "ham"]
    for name, fpr in FPR_TARGETS.items():
        point = operating_point(scores, labels, threshold_at_fpr(ham_scores, fpr))
        metrics |= {
            f"threshold_at_{name}_fpr": point.threshold,
            f"recall_at_{name}_fpr": point.recall,
            f"ham_fpr_at_{name}_fpr": point.ham_fpr,
            f"spam_fpr_at_{name}_fpr": point.spam_fpr,
        }
    return metrics
